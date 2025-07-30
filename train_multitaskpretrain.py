"""
TODO:
* 去掉 CLIP 的 image encoder，✅
* 将 vae 的 lantent 改为使用 info_mask 控制 ✅
* 加入 text_drop 的使用  ✅
* 加入 loss_mask 控制 ✅
* 修改 log 内容 ✅
* 加入 T5 encoder, 不同文本编码过 MLP ✅
* 除去 unet 中的 image cross-attention ✅
* 加入 时序层的 transformer ✅
* 修改 unet 中的结构控制嵌入方式 ✅
* 控制 pred_mask 来使用不同关键帧的数据 ✅
* 收集统计更多的数据
* 加入 bf16 训练 
* 尝试减小维度 或 减少某些层来降低显存占用 ✅ 
* 迁移到南京 ✅
* 多机多卡 ✅
* 软课程学习 ✅
* 重新确定课程学习从易到难模式 ✅
* GPT 4o 数据更新
* 开始训练后自动启动 tensorboard 8080 ✅
"""

import argparse
import logging
import math
import io
import os
import sys
import random
from pathlib import Path
import gpustat

import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import safetensors
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, T5EncoderModel
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

from transformers.utils import ContextManagers

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)
from vldm.DFGN import DFGN

from diffusers.schedulers import DDIMScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from lmdb_dataset_v2 import DatasetForMultitaskPretrain as Dataset

import ipdb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3950"

pwd = os.path.dirname(os.path.realpath(__file__))
logger = get_logger(__name__, log_level="DEBUG")


class StreamToLogger(object):
    def __init__(self, logger, level) -> None:
        self.logger = logger
        self.level = level
        self.linebfu = ""

    def write(self, buf: str):
        self.logger.log(self.level, buf)

    def flush(self):
        pass

class LogGpuStatus():
    """
    Control the GPU out based on the out frequency
    """
    def __init__(self, thre = 20, gap = 100):
        self.count = 0 
        self.thre = thre
        self.gap = gap
        
    def out_to_log(self, info):
        if self.count < self.thre:
            self.log_gpu(info)
        elif self.count % self.gap == 1:
                self.log_gpu(info)
        self.count += 1

    def log_gpu(self, info):
        state = gpustat.new_query()
        mem_f = io.StringIO()
        state.print_formatted(fp=mem_f, show_cmd=True, show_user=True, show_pid=True)
        logger.info(f"\n{info}: {mem_f.getvalue()}")


def cnt_net_params(net):
    count_param = []
    for name, params in net.named_parameters():
        if params.requires_grad == True:
            count_param.append(params)
    net_params = sum(x.numel() for x in count_param)
    logger.info(f"\nParamas: {round(net_params / (1000 * 1000 * 1000), 3)} B")


class TaskRandomChosen():
    def __init__(self, total_curriculum_learning_step=500000):
        # condition map
        self.tasks = [
            'i2v',
            't2v',
            'it2v',
            'interpolation',
            'pre_frame_prediction',
            'post_frame_prediction',
        ]

        self.task_weight = {
            'i2v': [0.1, 0.5, 0.9],
            't2v': [0.1, 0.5, 0.9],
            'it2v': [0.1, 0.5, 0.9],
            'interpolation': [0.9, 0.5, 0.1],
            'pre_frame_prediction': [0.7, 0.5, 0.3],
            'post_frame_prediction': [0.7, 0.5, 0.3]
        }
        
        self.total_curriculum_learning_step = total_curriculum_learning_step
        self.loss_batch = []
        self.K_p, self.K_i, self.K_d = 5, 3, 1.3
        self.P_a_batch = []
        self.P_ratio_map = {
            0: {
                'i2v': 0.1,
                't2v': 0.1,
                'it2v': 0.1,
                'interpolation': 0.9,
                'pre_frame_prediction': 0.5,
                'post_frame_prediction': 0.5
            },
            1: {
                'i2v': 0.3,
                't2v': 0.3,
                'it2v': 0.3,
                'interpolation': 0.7,
                'pre_frame_prediction': 0.5,
                'post_frame_prediction': 0.5
            },
            2: {
                'i2v': 0.5,
                't2v': 0.5,
                'it2v': 0.5,
                'interpolation': 0.5,
                'pre_frame_prediction': 0.5,
                'post_frame_prediction': 0.5
            },
            3: {
                'i2v': 0.7,
                't2v': 0.7,
                'it2v': 0.7,
                'interpolation': 0.3,
                'pre_frame_prediction': 0.5,
                'post_frame_prediction': 0.5
            },  
            4: {
                'i2v': 0.9,
                't2v': 0.9,
                'it2v': 0.9,
                'interpolation': 0.1,
                'pre_frame_prediction': 0.5,
                'post_frame_prediction': 0.5
            },              
        }
    
    def log_loss(self, cur_loss):
        if len(self.loss_batch) < 1000:
            self.loss_batch.append(cur_loss)
        else:
             self.loss_batch =  self.loss_batch[1:] + [cur_loss]
        return None

    def return_weights_base_step_pid(self, global_step):
        ratio = global_step / self.total_curriculum_learning_step 
        
        # static
        
        if global_step < (self.curriculum_learning_end // 2):
            ratio = global_step / (self.curriculum_learning_end // 2)
            cur_task_weight_map = {}
            for k,v in self.task_weight.items():
                cur_task_weight_map[k] = (v[1] - v[0]) * ratio + v[0]
        elif global_step < self.curriculum_learning_end :
            ratio = (global_step - (self.curriculum_learning_end // 2)) / (self.curriculum_learning_end // 2)
            cur_task_weight_map = {}
            for k,v in self.task_weight.items():
                cur_task_weight_map[k] = (v[2] - v[1]) * ratio + v[1]
        else:
            cur_task_weight_map = {}
            for k,v in self.task_weight.items():
                cur_task_weight_map[k] = v[2]

        # adaptive
        # Bug 修复: 将中文括号 `（` 和 `）` 替换为英文括号 `(` 和 `)`
        P_a = self.K_p * self.loss_batch[-1] + self.K_i * (sum(self.loss_batch) / len(self.loss_batch)) + self.K_d * (loss[-1] - loss[-2])
        
        if len(self.P_a_batch) >= 10:
            for idx in range(len(self.P_a_batch)-1):
                if P_a > self.P_a_batch[idx+1]:
                    break
            ratio = idx / len(self.P_a_batch)
        else:
            ratio = 1
        ratio = int(ratio * 3)
            
        if len(self.P_a_batch) < 1000:
            self.P_a_batch.append(P_a)
        else:
             self.P_a_batch =  self.P_a_batch[1:] + [P_a]        
        self.P_a_batch.sort()
    
        adaptive_task_weight_map = self.P_ratio_map[ratio]
        
        return {k:cur_task_weight_map[k]+adaptive_task_weight_map[k] for k in cur_task_weight_map.keys()}
    
    # def return_weights_base_step(self, global_step):
    #     ratio = global_step / self.total_curriculum_learning_step 
    #     interpolation =  (1 + math.cos(min(ratio, 0.75) * 2* math.pi)) / 2
    #     prediction = (1 + math.sin((min(ratio, 0.75) - 0.25 ) * 2 * math.pi   )) / 2 * 0.5 + 0.25
    #     generation = (1 + math.sin((min(ratio, 0.75) - 0.25 ) * 2 * math.pi   )) / 2
        
    #     return {
    #         'i2v': generation,
    #         't2v': generation,
    #         'it2v': generation,
    #         'interpolation': interpolation,
    #         'pre_frame_prediction': prediction,
    #         'post_frame_prediction': prediction
    #     }
        
        
    def make_condition(self, task, min_num_frames):
        image_condition = [0, 0, 0, 0, 0, 0, 0, 0]
        if task == 'i2v':
            text_drop = True
            image_condition[random.randint(0, min_num_frames-1)] = 1
        elif task == 't2v':
            text_drop = False
        elif task == 'it2v':
            text_drop = False
            image_condition[random.randint(0, min_num_frames-1)] = 1
        elif task == 'interpolation':
            text_drop = False
            image_condition = random.choice([
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 1, 0],
            ])
        elif task == 'pre_frame_prediction':
            text_drop = False
            start = random.randint(1, int(min_num_frames/2)+1)
            image_condition[start:] = [1] * (len(image_condition) - start)
        elif task == 'post_frame_prediction':
            text_drop = False
            end = random.randint(int(min_num_frames/2), min_num_frames-1)
            image_condition[:end] = [1] * end
        else:
            raise Exception('Not Implementation.')
        return {
            'text_drop': text_drop,
            'info_mask': image_condition
        }

    def get_task_condition(self, min_num_frames, steps, task=None):
        if task == None:
            task_weight_map = self.return_weights_base_step(steps)        
            tasks, weights = zip(*task_weight_map.items())
            task = random.choices(self.tasks, weights=weights, k=1)[0]
        task_condition = self.make_condition(task, min_num_frames)
        return task, task_condition  

    def get_masked_info(self, info, task_mask, frame_mask):
        """
            info: torch.Size([1, 4, 8, 32, 32]) [b, c, f, h, w]
            task_mask: [1, 0, 0, 0, 0, 0, 0, 1]
            frame_mask: [[1, 1, 1, 1, 0, 0, 0, 0]]
        """
        # info mask
        task_mask = torch.tensor(task_mask)
        task_mask = task_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(1).to(info.device)
        info = info * task_mask

        # frame mask
        frame_mask = torch.tensor(frame_mask)
        frame_mask = frame_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).to(info.device)
        info = info * frame_mask

        return info

    def get_masked_loss(self, info, frame_mask):
        """
            info: torch.Size([1, 4, 4, 32, 32]) [b, c, f, h, w]
            frame_mask: [[1, 1, 1, 1, 0, 0, 0, 0]]
        """
        # frame mask
        frame_mask = torch.tensor(frame_mask)
        frame_mask = frame_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).to(info.device)
        info = info * frame_mask

        return info

gpu_viewer = LogGpuStatus()

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help=(
            "The json file of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). "
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=6666, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default) `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--machine_rank", type=int, default=0, help="For distributed training: machine_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--init_from_pretrained_2d",
        type=str,
        default=None,
        help=(
            "Whether to init the model from a pretrained unet2d based SD model."
        )
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2video-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/log.txt"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help=(
            "Image dirs."
        ),
    )
    

    args = parser.parse_args()
    # return args

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=args.log_path,
        filemode='w',
    )

    # 定义一个输出到console的handler，并插入到logging的Logger中去
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(filename)s %(funcName)s %(lineno)s: %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %A %H:%M:%S",
        )
    )
    # 将这个额外定义的handler注册到loggor中去
    one_logger = logging.getLogger(__name__)
    one_logger.addHandler(console)  # 实例化添加handler

    sys.stdout = StreamToLogger(one_logger, logging.INFO)
    sys.stderr = StreamToLogger(one_logger, logging.ERROR)

    gpu_viewer.out_to_log('GPU Usage:')

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed now.
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(os.path.join(args.init_from_pretrained_2d, 'scheduler'))

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):

        vae = AutoencoderKL.from_pretrained(
            args.init_from_pretrained_2d, subfolder="vae", 
        )
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        # image and text tokenlizer
        # clip_tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.init_from_pretrained_2d, 'tokenizer'))
        T5_tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.init_from_pretrained_2d, 'models--uer--t5-base-chinese-cluecorpussmall/snapshots/206f3951d77e7faffaef6806a33fc0eae267b6e3'))

        
        clip_model_path = os.path.join(args.init_from_pretrained_2d, 'cn_clip/clip_cn_vit-h-14.pt')
        clip_model_name, clip_model_input_resolution = "ViT-H-14@RoBERTa-wwm-ext-large-chinese", 224
        with open(clip_model_path, 'rb') as opened_file:
            # loading saved checkpoint
            checkpoint = torch.load(opened_file, map_location="cpu")
        cn_clip_model = clip.utils.create_model(clip_model_name, checkpoint)
        cn_clip_model.float()
        PAD_INDEX = cn_clip_model.tokenizer.vocab['[PAD]']
        # cn_clip_model = CLIPTextModel.from_pretrained(args.init_from_pretrained_2d, subfolder="text_encoder")
        T5_model = T5EncoderModel.from_pretrained(os.path.join(args.init_from_pretrained_2d, 'models--uer--t5-base-chinese-cluecorpussmall/snapshots/206f3951d77e7faffaef6806a33fc0eae267b6e3'))

    state_dict = safetensors.torch.load_file(os.path.join(args.init_from_pretrained_2d, 'unet/diffusion_pytorch_model.safetensors'), device="cpu")
    unet = DFGN()

    # miss, unexp = unet.load_state_dict(state_dict, strict=False)

    # for name, params in unet.named_parameters():
    #     if name not in miss and name not in unexp:
    #         params.requires_grad = False
    #     else:
    #         params.requires_grad = True

    cnt_net_params(unet)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    cn_clip_model.requires_grad_(False)
    T5_model.requires_grad_(False)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DFGN.from_pretrained(os.path.join(input_dir, "unet"))
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = Dataset(
        args.dataset_file, 
        args.resolution,
        image_dir = args.image_dir
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    
    def tokenize_captions(captions, is_train=True):
        inputs = clip.tokenize(captions, context_length = 256)
        attn_mask = inputs.ne(PAD_INDEX)
        return inputs, attn_mask

    def T5_tokenize_captions(captions, is_train=True):
        inputs = T5_tokenizer(
            text=captions, 
            padding="max_length", 
            max_length=256,
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids

    def collate_fn(examples):
        tensors = torch.stack([example["tensor"] for example in examples])
        tensors = tensors.to(memory_format=torch.contiguous_format).float()

        input_ids, attn_mask = tokenize_captions([example["caption"] for example in examples])
        T5_input_ids = T5_tokenize_captions([example["caption"] for example in examples])

        frame_mask = [example['frame_mask'] for example in examples]

        return {"tensors": tensors, "clip_input_ids": input_ids, "attn_mask": attn_mask, "T5_input_ids": T5_input_ids, "frame_mask": frame_mask}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"Training with {weight_dtype}")

    # Move text_encode and vae to gpu and cast to weight_dtype
    cn_clip_model.to(accelerator.device, dtype=weight_dtype)
    T5_model.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if 'checkpoint' in args.resume_from_checkpoint:
            path = args.resume_from_checkpoint
        elif args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        elif path == args.resume_from_checkpoint:
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(path, map_location="cpu")
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # 多任务随机选择
    trc = TaskRandomChosen(total_curriculum_learning_step=args.max_train_steps)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        if args.resume_from_checkpoint and epoch == first_epoch:
            curr_train_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)
            logger.info(f'Skipped {resume_step} batches in the {first_epoch} epoch')
            progress_bar.update(resume_step//args.gradient_accumulation_steps)
        else:
            curr_train_dataloader = train_dataloader
        for step, batch in enumerate(curr_train_dataloader):

            with accelerator.accumulate(unet):
                # Convert images to latent space
                videos = batch["tensors"] # torch.Size([1, 3, 8, 224, 224])
                input_ids = batch["clip_input_ids"] # torch.Size([1, 77])
                T5_input_ids = batch["T5_input_ids"] # torch.Size([1, 77])
                frame_mask = batch["frame_mask"] # [[1, 1, 1, 1, 0, 0, 0, 0]]
                attn_mask = batch['attn_mask']

                # 得到当前训练任务
                cur_task, cur_condition = trc.get_task_condition(min([sum(m) for m in frame_mask]), global_step)
                text_drop, info_mask = cur_condition['text_drop'], cur_condition['info_mask']

                # 16, 3, 8, 256, 256
                b, c, f, h, w = videos.shape

                # 128, 3, 256, 256
                videos = videos.permute(0, 2, 1, 3, 4).reshape(b*f, c, h, w)

                # 128, 4, 32, 32
                latents = vae.encode(videos.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 16, 4, 8, 32, 32
                latents = latents[None, :].reshape((b, f, -1) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)

                # 3.3 Prepare additional conditions for the UNet.
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = cn_clip_model.bert(input_ids, attention_mask=attn_mask)[0] # torch.Size([2, 52, 1024])
                
                T5_encoder_hidden_states = T5_model(T5_input_ids)[0] # torch.Size([1, 77, 768])
                
                # 若 text_drop 为真，则丢弃文本控制
                if text_drop:
                    encoder_hidden_states = encoder_hidden_states * 0
                    T5_encoder_hidden_states = T5_encoder_hidden_states * 0

                # 制作被 mask 掉的结构控制条件
                image_latents = trc.get_masked_info(latents, info_mask, frame_mask)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    clip_hidden_states = encoder_hidden_states,
                    t5_hidden_states = T5_encoder_hidden_states,
                    image_latents = image_latents,
                    ).sample  # torch.Size([1, 4, 8, 28, 28])

                # loss mask
                model_pred = trc.get_masked_loss(model_pred, frame_mask)
                target = trc.get_masked_loss(target, frame_mask)

                # calc loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                gpu_viewer.out_to_log('GPU Usage:')
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                gpu_viewer.out_to_log('GPU Usage:')

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"cur_task": cur_task, "train_loss": train_loss}, step=global_step)
                trc.log_loss(train_loss)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "cur_task": cur_task}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()