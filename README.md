仓库名称：RDTF
项目简介：RDTF 是一个用于生成基于文本的动态图像的项目。它使用了 i2vgen 模型和 lora 等技术进行微调，以生成具有动态效果的图像。

训练方式：lora 微调

训练：bash shells/train_multitaskpretrain.sh

调用：/usr/local/envs/diffusers/bin/python examples_lora.py

****
效果展示：

1. 基于 i2v 的表情生成样例

| 静态图像 | 动态化示例 | 静态图像 | 动态化示例 | 
| ------------- | ------------- | ------------- | ------------- | 
| ![image](assets/i2v/a.png)  | ![image](assets/i2v/a.gif)    | ![image](assets/i2v/b.png)  | ![image](assets/i2v/b.gif) | 
| ![image](assets/i2v/c.png)  | ![image](assets/i2v/c.gif)    | ![image](assets/i2v/d.png)  | ![image](assets/i2v/d.gif) | 
| ![image](assets/i2v/e.png)  | ![image](assets/i2v/e.gif)    | ![image](assets/i2v/f.png)  | ![image](assets/i2v/f.gif) | 


2. 基于 it2v 的关键帧生成样例

| 文本描述 | 静态图像 | 真值结果 | 生成结果 | 
| ------------- | ------------- | ------------- | ------------- | 
| 图片上有一个白色的卡通小人物，可能是一个婴儿或一个孩子，有一个可爱的笑脸。这个角色似乎是伸出舌头，给人一种俏皮可爱的印象。这个角色也被描述为有一个黑色的头，增加了它可爱的外观。总的来说，这个形象给人一种幸福和娱乐的感觉，因为这个角色看起来很享受。  | ![image](assets/it2v/0a664dd5572c1afa45cc6e340e63232d.png)    | ![image](assets/it2v/0a664dd5572c1afa45cc6e340e63232d_gt.gif)  | ![image](assets/it2v/0a664dd5572c1afa45cc6e340e63232d_pred.gif) | 
| 图中有一只白色和棕色的小狗站在一个房间里。这只狗似乎在微笑，这给了它一个可爱和友好的表情。狗的表情所传达的情感是快乐的，可爱的。至于动作，狗是站着不动，看着观众，邀请一个积极的互动或只是享受这一刻。  | ![image](assets/it2v/00b4a3496aa1db61ee04f1c43a14a915.png)    | ![image](assets/it2v/00b4a3496aa1db61ee04f1c43a14a915_gt.gif)  | ![image](assets/it2v/00b4a3496aa1db61ee04f1c43a14a915_pred.gif) | 
| 这张图片上有一个看起来很悲伤的白色卡通人物，或者是一个孩子抱着一颗破碎的心哭泣的漫画风格的插图。这个角色似乎正经历着情感上的痛苦，可能是因为一个破碎的玩具或者一颗破碎的心。这个场景唤起一种同理心和情感脆弱的感觉。  | ![image](assets/it2v/0c3eb8bb0379cc1ac3f22d2c46580254.png)    | ![image](assets/it2v/0c3eb8bb0379cc1ac3f22d2c46580254_gt.gif)  | ![image](assets/it2v/0c3eb8bb0379cc1ac3f22d2c46580254_pred.gif) | 
| 在图片中，有一个白色的，卡通般的黑眼睛的猪站在一个暴力的姿势。这只猪看起来怒目而视，给人一种侵略和愤怒的印象。猪脸上的表情暗示着一种危险或潜在伤害的感觉。这张图片可能会引起观众的警惕或不安，因为这头猪似乎在展示性的行为。  | ![image](assets/it2v/1bc8bd9788b9dcb2bc3ccea65ecf9686.png)    | ![image](assets/it2v/1bc8bd9788b9dcb2bc3ccea65ecf9686_gt.gif)  | ![image](assets/it2v/1bc8bd9788b9dcb2bc3ccea65ecf9686_pred.gif) | 
| 这张图片上的人物坐在一个凌乱的架子上，打着领带，手里拿着一本书。角色坐在架子上时显得很惊讶或猝不及防。场景设置在一个房间里，周围散落着各种书籍，营造出一种好奇心和有趣的环境。  | ![image](assets/it2v/1f996103520f43c2a4d96e89c8e58d00.png)    | ![image](assets/it2v/1f996103520f43c2a4d96e89c8e58d00_gt.gif)  | ![image](assets/it2v/1f996103520f43c2a4d96e89c8e58d00_pred.gif) | 
| 照片上是一个微笑的亚洲男婴，可能是日本人，穿着和服。他双手合十，似乎在享受幸福和满足的时刻。宝宝欢快的表情和手势营造出温馨可爱的场景。  | ![image](assets/it2v/2ab8d36c66ac16a714324753b5b9a659.png)    | ![image](assets/it2v/2ab8d36c66ac16a714324753b5b9a659_gt.gif)  | ![image](assets/it2v/2ab8d36c66ac16a714324753b5b9a659_pred.gif) | 
| 这张图片的特点是一张白色的圆嘴，里面有一个音符，类似于一张脸。插图简单而卡通，音符放在嘴巴附近，就像是在唱歌或演奏。整体场景唤起一种俏皮和创造性的氛围。  | ![image](assets/it2v/2d452f8257359c9477c167fc3aa0545b.png)    | ![image](assets/it2v/2d452f8257359c9477c167fc3aa0545b_gt.gif)  | ![image](assets/it2v/2d452f8257359c9477c167fc3aa0545b_pred.gif) | 
| 这张图片的主角是一只可爱的、白色和棕色相间的兔子，长着一双富有表情的眼睛。这只兔子似乎很享受，因为它的腿上抱着一个小而圆的食物，很可能是一块饼干。这一幕捕捉到了兔子享受美食时的满足和喜悦。  | ![image](assets/it2v/2dbe416bc1d669ac50b26234e504d818.png)    | ![image](assets/it2v/2dbe416bc1d669ac50b26234e504d818_gt.gif)  | ![image](assets/it2v/2dbe416bc1d669ac50b26234e504d818_pred.gif) | 
| 这张照片的主角是一只看起来很悲伤的白兔子，有着粉红色的耳朵和眼睛。它似乎是站在一个平面上，如一张纸或一个计算机图形背景。兔子悲伤的表情和粉红色的细节在观众中创造了一种同理心和温暖的感觉。  | ![image](assets/it2v/3d8a8885092e2c420c4f2dc6092c9cef.png)    | ![image](assets/it2v/3d8a8885092e2c420c4f2dc6092c9cef_gt.gif)  | ![image](assets/it2v/3d8a8885092e2c420c4f2dc6092c9cef_pred.gif) | 
| 这幅图像的特征是一个红色的生物，可能是一个恶魔或红脸怪物，有角和锋利的牙齿。它看起来像是在炫耀它的角，并处于皱眉或愤怒的情绪中。生物似乎是图像的主要对象。  | ![image](assets/it2v/3eb3986237bbe10ee0773dfd0cec4936.png)    | ![image](assets/it2v/3eb3986237bbe10ee0773dfd0cec4936_gt.gif)  | ![image](assets/it2v/3eb3986237bbe10ee0773dfd0cec4936_pred.gif) | 

3. 插帧生成样例子

| 低帧率 GIF | 插帧后 GIF | 真值 GIF |
|-----|-----|-----|
| ![image](assets/frame/1cd94771ada4add299d4f344e1d8f768_src.gif) | <img width="" src="assets/frame/1cd94771ada4add299d4f344e1d8f768_pred.gif" alt="1cd94771ada4add299d4f344e1d8f768_pred.gif" /> | <img width="" src="assets/frame/1cd94771ada4add299d4f344e1d8f768_gt.gif" alt="1cd94771ada4add299d4f344e1d8f768_gt.gif" /> |
| <img width="" src="assets/frame/1d1dbeeb672051072aa874b24f03e759_src.gif" alt="1d1dbeeb672051072aa874b24f03e759_src.gif" /> | <img width="" src="assets/frame/1d1dbeeb672051072aa874b24f03e759_pred.gif" alt="1d1dbeeb672051072aa874b24f03e759_pred.gif" /> | <img width="" src="assets/frame/1d1dbeeb672051072aa874b24f03e759_gt.gif" alt="1d1dbeeb672051072aa874b24f03e759_gt.gif" /> |
| <img width="" src="assets/frame/3e3280c797e219775a5418f5448f784d_src.gif" alt="3e3280c797e219775a5418f5448f784d_src.gif" /> | <img width="" src="assets/frame/3e3280c797e219775a5418f5448f784d_pred.gif" alt="3e3280c797e219775a5418f5448f784d_pred.gif" /> | <img width="" src="assets/frame/3e3280c797e219775a5418f5448f784d_gt.gif" alt="3e3280c797e219775a5418f5448f784d_gt.gif" /> |
| <img width="" src="assets/frame/4f59e11e2220591052d5fee2b611786c_src.gif" alt="4f59e11e2220591052d5fee2b611786c_src.gif" /> | <img width="" src="assets/frame/4f59e11e2220591052d5fee2b611786c_pred.gif" alt="4f59e11e2220591052d5fee2b611786c_pred.gif" /> | <img width="" src="assets/frame/4f59e11e2220591052d5fee2b611786c_gt.gif" alt="4f59e11e2220591052d5fee2b611786c_gt.gif" /> |


****
