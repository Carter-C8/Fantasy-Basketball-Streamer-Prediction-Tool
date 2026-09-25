<div align="center">
  
# Fantasy-Basketball-Streamer-Prediction-Tool

***

A full-stack ML application for fantasy basketball

<img width="500" height="500" alt="fantasy-basketball-badge-1" src="https://github.com/user-attachments/assets/cd011bed-9668-4fe0-8914-cf59e1200bcf" />

***

## Overview
<div align = "left">
  
This tool is focused on helping Fantasy Basketball managers **find the best streaming candidates** to help win their ESPN Fantasy Basketball Leagues. NBA 'streamers' are players who are short-term players added from the waiver wire, with a focus on maximizing total games played using non-star players that have short-term potential. This tool is a **full-stack** application that leverages a **machine learning model** to identify players who will generate the most fantasy points for the week. The model **predicts potential fantasy points** for every NBA player and allows a user to connect the tool to their **personal ESPN Fantasy League** to filter out non-available players (players already on other fantasy teams) who will not on the waiver-wire.

</div>

## Application Preview

<div align="left">

<h3>I. Home Page</h3>

</div>

<div align = "left">
  <kbd>
<img width="1844" height="1844" alt="043EA6EB-6D34-428F-8189-359C01A837FC_1_201_a" src="https://github.com/user-attachments/assets/89b590c2-bc5e-487f-ac1b-4b91073a107d" />
</kbd>
  
The Home Page features a scrollable list of every NBA player, from most predicted fantasy points to least predicted fantasy points. For **every player, their headshot, weekly predicted fantasy points, name, team, and games scheduled for the week are displayed**. Users can also click on a player to go to their player profile and see more detailed information.<br>

***

<div align="left">

<h3>II. Player Filters</h3>

</div>
<kbd>
<img width="367" height="52" alt="Screenshot 2026-09-23 at 12 06 25 AM" src="https://github.com/user-attachments/assets/dc0c693a-d06b-40ff-b813-9794666675b1" />
<img width="237" height="62" alt="Screenshot 2026-09-23 at 12 06 34 AM" src="https://github.com/user-attachments/assets/a79230c3-c327-4c29-b9cf-815dbdbbb150" /><br>
</kbd>

There are also buttons to **filter players by position**, and predictions by the **current week** and the **next week**. These features help users find players to help fill the necessary position in their team, and also help plan ahead by looking into next week's matchup.<br>

***

<div align="left">

<h3>III. ESPN Fantasy League Functionality</h3>

</div>

<kbd>
  <img width="821" height="335" alt="Screenshot 2026-09-23 at 12 08 02 AM" src="https://github.com/user-attachments/assets/acf1fd47-b057-4f4c-bb82-2b31ed35f5ae" />
</kbd>

Users can connect this tool to their own personal **ESPN Fantasy League**, allowing a user to **filter out all non-available players** (players who are already rostered on a team in the fantasy league). This allows the user to save time by only looking at players who will be available to them. This tool is also focused on predicting the fantasy points and ranking NBA streamers, meaning that the accuracy and testing was tuned towards these non-star players who will not be rostered on a fantasy league team.

***

<div align="left">

<h3>IV. Player Profile</h3>

</div>
<kbd>
<img width="704" height="486" alt="Screenshot 2026-09-23 at 12 08 41 AM" src="https://github.com/user-attachments/assets/1dc8a755-6e0e-48b3-8c07-4f267b425015" />
</kbd>

The player profile contains the player's average fantasy points per game over the past 10 games. Additionally, that player's **scheduled games for the week are shown**, with the opponent and **predicted fantasy points for that specific game** are displayed.

***

<kbd>
<img width="677" height="161" alt="Screenshot 2026-09-23 at 12 08 57 AM" src="https://github.com/user-attachments/assets/01013269-c61e-4774-a842-359d67e523a0" />
  </kbd>
  
Clicking on a game in the player profile will bring up more detailed statistics for that player such as their average minutes, points, rebounds, assists, steals, and blocks over their past 4 games.

***

<div align="center">
  
## Core Features

<div align = "left">
  
### Machine Learning Model
  
→ **Historical Data Insights**: _Analyze historical data over the last five seasons to predict fantasy point generation. Training, Test, and Validation sets filtered out players who had top-156 fantasy point generation, allowing the model to focus on minute details for streamer potential._ <br>

→ **Data Preparation & Preprocessing**: _Designed a **preprocessing pipeline** to clean data and create useful categories for the model to learn from such as average fantasy points over past 10 games, average minutes over past 3 games, opponent defensive rating, opponent offensive rating, and team available minutes._ <br>

→ **Rigorous Testing & Evaluation**: _Using the 2025-2026 as the testing set, the model was able to be evaluated on the most recent NBA season with the most up-to-date trends and style of play. The model was **tested and compared against a Baseline**, which merely picked the available player averaging the most fantasy points over the past 10 games, and was able to beat the baseline by 209 points over a test season. Used paired significance testing (t-test and Wilcoxon signed-rank, checked for agreement) and bootstrap confidence intervals to check for statistical significance._<br>

### Automated Pipeline
→ **Automated Shell Script**: _This project is able to be **run automatically** to retrieve data every day during the NBA season, preprocess that data, get the model's predictions, and **update the database** and website with the new predictions._ <br>

→ **Model Retraining \*in progress**: _Use new data from the current NBA season to **automatically retrain the model**. Automatically test and evaluate the model to check for improvements before classifying the model as ready for deployment._<br>

### Full-Stack Functionality
→ **UX/UI**: _React interface with multiple different pages for the user to interact with and for data to be displayed._
</div>

***
## Project Details

### Development Progress:

| Component    | Progress |
| -------- | ------- |
| App Interface (Full-Stack) | ✅ Completed UI/UX Design Implementation  |
| Machine Learning Prediction Model | ✅ Completed Data Pipeline, Model Training, and Evaluation |
| Cloud Functionality | ⚠️ In Progress |

| **Current Task** |
| -------- |
| Deploy the application on AWS to allow for cloud functionality/use without local deployment setup |

<br>
