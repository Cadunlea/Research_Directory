# Research_Directory
Research materials for Eating Trajectory study stored here

---

## NOTEBOOK

### 25 August 2026 | 3:58 pm - 8:12 pm
**Primary objective:** Create slides, update GitHub documentation, test scripts, and build hyperparameter tuning pipeline

My primary objective for the day was to get some testing done, now that I had access to the computer. In the middle of this, I found out that my github documentation was on the wrong account. I had to restart it from scratch, completely scrapping the other account's documents for consistency.
Below is a description of how that hyperparameter tuning testing worked. It took a long hours to run in the background. My goal was to obtain graphs I could use for the slides. As I waited for the script to finish, I created documentation for the github and created the slides that I will use for tomorrow's meeting.

* **Created Hyperparameter Sweep Script:** Imports the baseline training script as a module, reusing dataset construction, FFT feature extraction, train/val/test splits, and threshold-sweep evaluation functions.
* **Parameterized Model Architecture:** Implemented `ParamFrequencyCNN` and `train_model_with_params` to accept variable arguments (learning rate, batch size, dropout, channel widths, and positive-class weight scale) rather than using hardcoded values.
* **Fixed Data Split:** Pre-computes the train/val/test participant split once at startup so all grid search configurations train and validate on identical data.
* **Grid Search & Multi-Seed Runs:** Evaluates every hyperparameter combination across multiple random seeds to measure performance stability and account for weight initialization/batch shuffling noise.
* **Threshold Evaluation & Logging:** Tests models against the baseline threshold grid (0.30–0.90) for direct comparison. Progressively logs individual runs to CSV and aggregates mean/standard deviation metrics per configuration.

**25 August 2026, 8:55 am - 9:45 pm**  
**Primary objective:** Experiment with variations of the current model

Since I was not able to access the computer in the lab due to Kenzy using it, I worked with variations of the current model. I looked for potential bugs and adjusted parts of the code that Siavash asked for.

**Work Completed & Code Adjustments:**
* **Added Configurable label rule `compute_food_indicator()`**
  * Was hardcoded to the three-way OR; now a switchable rule (default "bout_or_bite")
  * **bout_or_bite:** food = (Chew Bout > 0) OR (Bite > 0)
  * **bout_only:** food = (Chew Bout > 0)
  * **bite_only:** food = (Bite > 0)
  * **count_or_bout_or_bite:** original rule, food = (Chew Count > 0) OR (Chew Bout > 0) OR (Bite > 0)
* **Added classification error rate in `metrics_from_probabilities()`**
  * `error_rate = 1 - accuracy`
* **Mean absolute error in `metrics_from_probabilities()`**
  * `mean_absolute_error = mean( |y_true_i - probability_i| )` over all i windows
  * This is the literal "absolute error," computed on the raw sigmoid probability output, not the thresholded 0/1 prediction so it's sensitive to how confident the model was.
* **Per-participant missing-data table `participant_diagnostics`, written to `participant_missing_fraction_report.csv`**
  * **Per participant/session:** `mean_missing_fraction` = mean over that session's kept windows of (fraction of samples equal to -1) plus span overlap, window counts, and whether it was excluded.
  * This was already printed to console in the original version. Now it is also collected into a table and written to CSV.

---

**23 August 2026, 4:05 pm - 6:44 pm**  
**Primary objective:** Work on a parameter tuning script prototype

I created a working prototype of parameter tuning to go along with the machine learning model previously made. At first, I thought about putting the parameter tuning inside of the ML script, but decided against it for convenience. I am not yet sure if it is compatible with the script, since it appeared that Kenzy was using the computer at this time. I also worked on learning how to create parameter tuning scripts, and how to possibly get the data into a csv.

**Work Completed:**
* Created parameter tuning script to go along with the current working model
* Investigated options for the parameter tuning script, such as which parameters to tune, turning the data into a csv, and where to put the tuning (in or out of the ML script)

---

**19 August 2026, 6:19 pm - 7:28 pm**  
**Primary objective:** Convert to Github, relearn structure of code

I worked on porting my changes to github, including this journal. I looked into methods I could use to port google slides over, but I could not find any way except by using a URL. For now, those will have to stay in the google drive that I created.  
Lastly, I relearned the purpose and the structure of the code, going over old notes and slides to do this.

**Work Completed:**
* Port changes to github and create a ReadMe (will add scripts later)
* Studied the structure of the machine learning algorithm
* Relearned terms, label rules, and design choices

---

**19 August 2026, 11:15 pm - 1:05 pm**  
**Primary objective:** Lab meeting, emails, RRSP contracts

I discussed my project ideas in our research meeting today. Additionally, I presented my slides for the August 19th 2026 lab meeting. The professor suggested that I move my documentation to GitHub.  
Lastly, I worked on emails to write to Dr. Sazonov.

**Work Completed:**
* Finished the first drafts phase 1 and phase 2 contracts
* Sent them to Dr. Sazonov for review, alongside my CWID
* Discussed scheduling and weekly hourly commitment

---

**18 August 2026, 9:41 pm - 11:35 pm**  
**Primary objective:** Research setup and housekeeping

I worked on the creation of the original notebook in google docs. Additionally, I worked on slides to present at tomorrow’s meeting (August 19th 2026). To have one home for my notebook and my presentations for the future, I created a google drive folder which will be shared with my instructors.  
Lastly, I worked on the draft of an email to write to Dr. Sazonov and Siavash. The contents of this email were focused on RRSP Contracts I and II.  
I worked on and off, with periods where I paused to do other things such as put my classes in my calendar. Worked in total for about an hour and a half.

**Work Completed:**
* Set up the structure of this notebook
* Wrote the first notebook entry
* Created a google drive folder to organize my research contents
* Drafted and later sent an email discussing Contract I and II for RRSP
