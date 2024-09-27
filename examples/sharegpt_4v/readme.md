# ShareGPT4V: Improving Large Multi-modal Models with Better Captions

[Paper](https://arxiv.org/pdf/2311.12793.pdf)

[Official Repo](https://github.com/ShareGPT4Omni/ShareGPT4V)

![Image](https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/teaser.png)


## Inference


1. Prepare weight files:

   According to the paper, there are three components of ShareGPT4V:

   a. language model - llama

   b. vision transformer - vit

   c. mlp that concats multi modal features - share4v model

   You can download the weight files through this [link](https://download-mindspore.osinfra.cn/toolkits/mindone/sharegpt_4v)

2. Edit the path:

   - edit the content in file `../sharegpt_4v/share4v/configs/config.json`

        ```
        "mm_vision_tower": "path/to/vit/configs",
        "mm_vision_tower_path":"/path/to/vit/weight",
        ```



3. Then you can run it *a.* using the run script or *b.* manually

   a.  using the run script

      ```
      cd ../examples/opensora_pku/scripts/sharegpt_4v
      ```

    -  edit the addresses in `run.sh` file: `/path/to/model/folder` and `/path/to/image/file`
    -  run the script file
      
      ```
      bash run.sh
      ```


    b. directly run by following command(under the folder `../examples/sharegpt_4v`):


        python share4v/eval/run_share4v.py --model-path "/path/to/model/folder" --image-file "/path/to/image/file"


   - optional: add the project into system path if needed


  # Experiment
  ## Environment

  Ascend 910B, CANN 7.3, MindSpore 2.3.1 official release

  ## Experiment Setting


  Graph/Dynamic mode: Dynamic

  Original weight file is ShareGPT4V_7B

  max_new_token: 1024

  query string: Describe this image


  ## Quantative Results

  |  |MindSpore|PyTorch|
  |--|--|--|
  |tokens/second|2.4|3.7|

  Bleu score: 0.17

  ## Experiment Results

<img src="https://github.com/user-attachments/assets/43071bfd-8dad-40ba-876d-729632bc58c9" alt="Sample Image" width="900" height="600">

  ||Genrated Caption|
  |--|--|
  |MindSpore|In the image, there are two young women engaged in a lively game of frisbee on a grassy field. The woman in the foreground, dressed in a vibrant red shirt and blue shorts, is in the midst of throwing the frisbee, her body leaning into the action as she reaches out with her right hand to make the throw. Her companion, clad in a yellow shirt and green shorts, is in the process of catching the frisbee, her body stretched out in anticipation. The frisbee, a white disc, is captured mid-air, frozen in the moment just before it reaches the woman in the red shirt. The field they're playing on is lush and green, a stark contrast to the clear blue sky above. In the distance, other players can be seen, their forms slightly blurred, adding depth to the scene. The woman in the yellow shirt is also wearing yellow shoes, matching her frisbee, while the woman in red is in black and white sneakers. The image is a snapshot of an active moment, full of energy and movement, capturing the spirit of the game. The frisbee, the central object of their attention, is white and appears to be made of plastic. The scene is set outdoors, with trees standing tall in the background, their leaves a mix of green hues, and a fence visible in the far distance. The image is a dynamic display of a frisbee game in progress, the players' positions suggesting a game in full swing. The woman in the yellow shirt is on the right, her body angled towards the frisbee, her focus intent clear. The frisbee, caught in the air, is the focal point of the image, its white color standing out against the greenery. The image is a dynamic tableau of a frisbee game in progress, the players' actions painting a picture of an active day outdoors.|
  |PyTorch|The image captures a dynamic moment in a frisbee game. Two young women are in the midst of an intense play. The woman on the left, clad in a vibrant red shirt and blue shorts, is diving towards the frisbee, her body stretched out in an impressive display of agility and determination. Her eyes are focused on the frisbee, which is suspended in mid-air between them. On the right, another woman is poised to intercept the frisbee. She's dressed in a contrasting yellow shirt and green shorts, her body angled towards the frisbee, ready to make a swift grab. The tension between the two players is palpable, a testament to the competitive spirit of the game. The background of the image reveals a lush green field, a perfect setting for an outdoor activity like frisbee. The field extends into the distance, providing a sense of depth to the image. The overall scene is a snapshot of a fun and active moment, encapsulating the spirit of sport and camaraderie.|

<img src="https://github.com/user-attachments/assets/25bacac6-5e44-42d7-b67f-73385ea82204" alt="Sample Image" width="900" height="600">

  ||Genrated Caption|
  |--|--|
  |MindSpore|The image captures a dynamic moment in a tennis match. A female tennis player, dressed in a white outfit, is in the midst of a powerful forehand shot. She is positioned on the right side of the court, her body leaning into the shot with her right arm extended, racket poised to strike the yellow tennis ball that's in mid-air. Her attire is a white dress and white shoes, and she's wearing a white headband, adding to the overall white theme of her outfit. The court beneath her feet is a vibrant green, contrasting with the green wall in the background. The ball is slightly to her right, indicating she's hitting it from the left side of the court. The background is a blur of motion, suggesting the speed of the game, but the focus is on her and the ball, emphasizing the intensity of the moment. The court is enclosed by a green fence, and there's a blurred figure in the distance, possibly a line judge or an opponent, adding depth to the scene. The image is a freeze-frame of action, a snapshot of a tennis match in progress. The player's stance and the position of the ball suggest a powerful return, a testament to the intensity of the game. The colors are vivid, with the green of the court and the white of her outfit and ball standing out against the blurred backdrop. The image is a testament to the player's skill and the fast-paced nature of the sport.|
  |PyTorch|In the image, a female tennis player is captured in the midst of a powerful forehand shot. She is dressed in a crisp white outfit, which contrasts sharply with the vibrant green of the tennis court beneath her feet. The tennis racket, gripped firmly in her right hand, is poised to strike the yellow tennis ball that hangs in the air to her right. The player's body is angled towards the ball, her left leg extended behind her in a classic tennis stance. Her right arm is fully extended, reaching out to meet the ball. The intensity of the game is palpable in her focused expression and the dynamic pose. In the background, a line judge can be seen, attentively observing the match. The green wall behind them provides a stark contrast to the player and the court, further emphasizing the action in the foreground. The image captures a single, frozen moment in the fast-paced game of tennis, highlighting the player's skill and concentration.|

<img src="https://github.com/user-attachments/assets/43f697e9-36a9-49e2-ae3f-085b4bfbd7cb" alt="Sample Image" width="900" height="600">

  ||Genrated Caption|
  |--|--|
  |MindSpore|In the image, a woman is seen paddleboarding on a serene lake, her body poised in a balanced stance as she navigates the calm waters. She's dressed in a blue shirt and black shorts, her attire contrasting with the vibrant blue of the paddleboard beneath her. The paddleboard, adorned with a red and white logo, cuts through the water, leaving a trail of ripples in its wake. The woman is holding a red paddle, using it to propel herself forward, her gaze directed towards the right side of the frame, perhaps anticipating her next move or simply enjoying the tranquility of the surroundings. The lake is nestled amidst a backdrop of lush green trees, their leaves rustling gently in the breeze. The sky above is a clear blue, mirroring the water below, creating a harmonious blend of nature's hues. The perspective of the image is from the side, giving a sense of her journey across the lake. The image captures a moment of peace and tranquility, a snapshot of an adventure in progress. However, the image itself is a testament to the joy of outdoor activities and the beauty of nature.|
  |PyTorch|In the image, a woman is seen paddleboarding on a serene lake. She is dressed in a blue shirt and black shorts, and is wearing a gray hat. The paddleboard she is standing on is blue and has a red paddle in her hands. The lake is surrounded by lush green trees, creating a beautiful backdrop for the scene. The sky above is clear and blue, adding to the tranquility of the setting. The woman appears to be in motion, gliding smoothly over the water. The image captures a moment of peace and enjoyment in nature.|
