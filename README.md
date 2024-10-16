# LCZ_neighborhood_generator
Script that creates a plan view of a city district according to local climate zone parameters. It can be used to create neighborhoods (with simplified building blocks similar e.g. to the superblocks in Barcelona)  in accordance to specific LCZ types and the exported image can be used as input to Envi-met.
The are only 4 parameters that have to be defined by the user at the top of the script: model area size. horizontal resolution, local climate zone and horizontal distance between roads.

Here is an example for a neighborhood with LCZ2 and a model area of 200x200 m2. Building heights of each block are generated randomly within the range defined by the LCZ:\
![image](https://github.com/user-attachments/assets/ad529f11-8a76-44ce-9c56-7981bece0086)

The sky view factor of the neighborhood can be diagnosed:\
![image](https://github.com/user-attachments/assets/0a79cf69-2ee5-4d29-b758-da8a2bc85313)

And the final input image for Envi-met Spaces can look like this e.g.:\
![labeled_height_grid](https://github.com/user-attachments/assets/e4b22cc6-a8aa-4600-9595-726596b0474f)
