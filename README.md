# Terraforming-Titans-WGC-Team-Optimizer
A solver for the WGC team optimizer.

## Instructions
Open the exe. It'll take a while to open, usually 10 seconds.

This is what you should see.

<img width="446" height="196" alt="image" src="https://github.com/user-attachments/assets/522d9eba-f402-470d-b341-34a64f432f8e" />

You can enter the levels of your facilities at the top.

Success chance is how likely you are to succeed at any given challenge (so a 100% chance means you are always guarenteed to succeed).

Because of math-reasons it's easier to calculate scientists when their levels are the same. Soldiers don't have this problem, however because I am a lazy developper please put your soldiers first, then your scientists to be able to input individual levels for your scientists.

Click calculate to find the optimal point distribution.

You can find the hazard approach and the maximum difficulty level in the top left.

<img width="446" height="196" alt="image" src="https://github.com/user-attachments/assets/670988c4-e21a-4c1e-9ae1-bd1ef295015c" />

Below that are the point distributions for each team member. Please note that because some team members have assigned points by default you may have to slightly tweak the 'auto' setting. In the image below, for example, I've had to set the 'auto' power to 21 to get the correct value of 20 from above.

<img width="517" height="243" alt="image" src="https://github.com/user-attachments/assets/8f1603d9-a45a-414b-9601-620ccff6e01c" />



## Disclaimer
This optimizer works best when you go soldier, natural scientist, social scientist. Having 2 soldiers is possible and works 99% of the time, however it may sometimes fail. Usually this can be solved by lowering the difficulty level by 1.

Additionally, sometimes the max level may go negative; this means that the optimizer cannot guarentee the team success chance. This generally only happens when you're below level 10, and in those scenarios, you'll just have to bite the bullet.
