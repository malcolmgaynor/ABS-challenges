import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="MLB ABS Challenge Decision Tool")
# header with my LinkedIn
hcol1, hcol2 = st.columns([4, 1])
with hcol1:
    st.title("MLB ABS Challenge Decision Tool")
    st.write("Created by Malcolm Gaynor")
with hcol2:
    st.markdown("""
    <div style="text-align: right; padding-top: 20px;">
        <a href="https://www.linkedin.com/in/malcolm-gaynor-531688225/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 width="30" height="30" style="margin-right: 10px;">
        </a>
    </div>
    """, unsafe_allow_html=True)

#st.title("MLB ABS Challenge Decision Tool")
#st.subheader("Should a player use an ABS challenge?")
#st.write("----")


challenge_tab, methodology_tab, data_tab, optimization_tab, classifiers_tab, limitations_tab = st.tabs([
    "Challenge Decisions",
    "Methodology",
    "Data",
    "Optimization Model",
    "ML Classifiers",
    "Limitations"
])

MODEL_PATH = Path(__file__).with_name("decision_tree_models.joblib")

@st.cache_resource
def load_models_and_data():
    return joblib.load(MODEL_PATH)

#read in the joblib decision tree data:
model_artifact = load_models_and_data()
tree_1_left = model_artifact["tree_1_left"]
tree_2_left = model_artifact["tree_2_left"]
feature_columns = model_artifact["feature_columns"]
class_names = model_artifact["class_names"]


tree_5_1_left = model_artifact["tree_5_1_left"]
tree_5_2_left = model_artifact["tree_5_2_left"]
logistic_1_left = model_artifact["logistic_1_left"]
logistic_2_left = model_artifact["logistic_2_left"]
hist_gb_1_left = model_artifact["hist_gb_1_left"]
hist_gb_2_left = model_artifact["hist_gb_2_left"]
hist_gb_1_left_mean_importance = model_artifact["hist_gb_1_left_mean_importance"]
hist_gb_2_left_mean_importance = model_artifact["hist_gb_2_left_mean_importance"]

hitterwoba = pd.read_csv("woba_stats.csv")
pitcherwoba = pd.read_csv("pitcher_woba_stats.csv")

#remove last_name, first_name = AVERAGE
hitterwoba = hitterwoba[hitterwoba["last_name, first_name"] != "AVERAGE"]
pitcherwoba = pitcherwoba[pitcherwoba["last_name, first_name"] != "AVERAGE"]

#only keep name and woba columns
hitterwoba = hitterwoba[["last_name, first_name", "woba"]]
pitcherwoba = pitcherwoba[["last_name, first_name", "woba"]]

with challenge_tab:
    st.subheader("Should a player use an ABS challenge?")
    st.write("Always challenge if the call is obviously incorrect. This tool is just used to determine if it is worth using a challenge on a borderline call.")
    st.write("----")
    st.subheader("Input situation (before the pitch is thrown)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Baserunners")
        input_on_1b = st.checkbox("Runner on 1st?")
        input_on_2b = st.checkbox("Runner on 2nd?")
        input_on_3b = st.checkbox("Runner on 3rd?")
    with col2:
        st.write("Outs")
        input_outs_when_up = st.radio("Number of outs", options=[0, 1, 2])
    with col3:
        st.write("Count")
        col3_balls, col3_strikes = st.columns(2)
        with col3_balls:
            input_balls = st.radio("Number of balls", options=[0, 1, 2, 3])
        with col3_strikes:
            input_strikes = st.radio("Number of strikes", options=[0, 1, 2])
    input_inning = st.select_slider("Inning", options=list(range(1, 10)))

    #league averages used both as number_input defaults and to scale user-entered stats
    league_average_woba = 0.316
    league_average_ops = 0.719
    league_average_era = 4.2
    league_average_fip = 4.0

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Hitter")
        dontrunyethitter = False
        woba_or_ops = st.selectbox("Select metric to measure hitters (or choose hitter from database)", options=["None (assume league average)","wOBA", "OPS","Choose hitter from database"])
        if woba_or_ops == "wOBA":
            input_batter_woba = st.number_input("Batter wOBA", min_value=0.0, max_value=2.0, value=league_average_woba, step=0.001, format="%.3f")
        elif woba_or_ops == "OPS":
            input_batter_ops = st.number_input("Batter OPS", min_value=0.0, max_value=5.0, value=league_average_ops, step=0.001, format="%.3f")
        elif woba_or_ops == "Choose hitter from database":
            input_batter_from_db = st.selectbox("Enter hitter name from database. Stats include qualified hitters from the 2026 season through August.", options = ['']+hitterwoba["last_name, first_name"].tolist())
            if input_batter_from_db != '':
                dontrunyethitter = False
                input_batter_woba = hitterwoba.loc[hitterwoba["last_name, first_name"] == input_batter_from_db, "woba"].values[0]
            else:
                dontrunyethitter = True
    with col5:
        st.subheader("Pitcher")
        era_or_fip = st.selectbox("Select metric to measure pitchers", options=["None (assume league average)","ERA", "FIP", "wOBA allowed"])
        dontrunyetpitcher = False
        if era_or_fip == "ERA":
            input_pitcher_era = st.number_input("Pitcher ERA", min_value=0.0, max_value=100.0, value = league_average_era, step=0.01)
        elif era_or_fip == "FIP":
            input_pitcher_fip = st.number_input("Pitcher FIP", min_value=0.0, max_value=100.0, value = league_average_fip, step=0.01)

        elif era_or_fip == "wOBA allowed":
            input_pitcher_woba_allowed = st.number_input("Pitcher wOBA allowed", min_value=0.0, max_value=2.0, value=league_average_woba, step=0.001)
        elif era_or_fip == "Choose pitcher from database":
            input_pitcher_from_db = st.selectbox("Enter pitcher name from database. Stats include qualified pitchers from the 2026 season through August.", options = ['']+pitcherwoba["last_name, first_name"].tolist())
            if input_pitcher_from_db != '':
                input_pitcher_woba_allowed = pitcherwoba.loc[pitcherwoba["last_name, first_name"] == input_pitcher_from_db, "woba"].values[0]
                dontrunyetpitcher = False
            else:
                dontrunyetpitcher = True

    st.subheader("Challenge details")
    col6, col7 = st.columns(2)
    with col6:
        challenge_player = st.selectbox("Who is considering challenging?", options=["Batter", "Catcher", "Pitcher"])
    with col7:
        challenges_left = st.radio("How many challenges are left?", options=[1, 2])
    input_original_strike = 1 if challenge_player == "Batter" else 0
    if challenge_player == "Pitcher":
        st.warning("PITCHERS SHOULD NOT CHALLENGE. LEAVE THAT TO THE CATCHER.")

    
    if woba_or_ops == "None (assume league average)":
        input_batter_woba_plus = 1
    elif woba_or_ops == "OPS":
        input_batter_woba_plus = input_batter_ops / league_average_ops
    else:
        if not dontrunyethitter:
            input_batter_woba_plus = input_batter_woba / league_average_woba

    if era_or_fip == "None (assume league average)":
        input_pitcher_woba_plus = 1
    elif era_or_fip == "ERA":
        input_pitcher_woba_plus = input_pitcher_era / league_average_era
    elif era_or_fip == "FIP":
        input_pitcher_woba_plus = input_pitcher_fip / league_average_fip
    else:
        if not dontrunyetpitcher:
            input_pitcher_woba_plus = input_pitcher_woba_allowed / league_average_woba


    input_risp = int(input_on_2b or input_on_3b)
    input_bases_loaded = int(input_on_1b and input_on_2b and input_on_3b)
    input_action_count = int(
        (input_strikes == 2 and input_original_strike == 1)
        or (input_balls == 3 and input_original_strike == 0)
    )

    st.subheader("Back-end model")
    model_chosen = st.selectbox("Select machine learning model", options=[
        "Decision Tree (depth of 3)",
        "Decision Tree (depth of 5)",
        "Logistic Regression",
        "Histogram Gradient Boosting",
    ])
    show_details = st.checkbox("Show model details/explanation on how they impact the result")
    if show_details:
        st.markdown("- Decision Tree of depth 3 is the most interpretable model, and also the most conservative. Only challenges late in the game, unless it is a great scoring chance.")
        st.markdown("- Decision Tree of depth 5 is slightly less interpretable, but allows for more nuanced challenge opportunities. Similar baseline ideas as the previous decision tree.")
        st.markdown("- Logistic Regression is a linear model. Unlike the decision trees, it considers everything about the situation, and specifically prioritizes the batter/pitcher quality and baserunners. It is also generally conservative.")
        st.markdown("- Histogram Gradient Boosting is the most complex model. It is the most aggressive, and performs the best overall (while still skewing slightly conservative), but is not interpretable. The inning is still the most important predictor overall.")

    st.write("----") 

    if challenge_player == "Pitcher": 
        st.warning("Choose either Batter or Catcher to calculate challenge decision. Pitchers should not challenge.")   

    if dontrunyetpitcher or dontrunyethitter:
        st.warning("Choose a hitter and pitcher (or input data manually) to calculate challenge decision.")   


    if challenge_player != "Pitcher" and not dontrunyethitter and not dontrunyetpitcher and st.button(
        "Should you challenge?",
        use_container_width=True,
    ):
        input_data = pd.DataFrame([[
            input_on_1b,
            input_on_2b,
            input_on_3b,
            input_outs_when_up,
            input_inning,
            input_balls,
            input_strikes,
            input_batter_woba_plus,
            input_pitcher_woba_plus,
            input_original_strike,
            input_risp,
            input_bases_loaded,
            input_action_count,
        ]], columns=feature_columns)

        models = {
            "Decision Tree (depth of 3)": (tree_1_left, tree_2_left),
            "Decision Tree (depth of 5)": (tree_5_1_left, tree_5_2_left),
            "Logistic Regression": (logistic_1_left, logistic_2_left),
            "Histogram Gradient Boosting": (hist_gb_1_left, hist_gb_2_left),
        }
        model_pair = models[model_chosen]
        prediction_model = model_pair[0] if challenges_left == 1 else model_pair[1]
        prediction = int(prediction_model.predict(input_data)[0])
        if prediction ==1:
            st.success("Challenge")
        elif prediction == 0: 
            st.error("Do not challenge")

with methodology_tab:
    st.subheader("Methodology")
    st.write("The goal is to create a framework for deciding when to challenge borderline pitches. So, we start with a sample set of each borderline pitch in the 2026 MLB season (through the end of August). For each pitch, we calculate the run value of the situation, the change in run value if there would be a successful challenge, and the probability of a challenge succeeding.")
    st.markdown("Then, considering complete hindsight (we know every challengeable pitch in the game) we start by building an optimization model (specifically optimizing using Backward Dynamic Programming with Bellman's Equations) to choose which pitches should be challenged. This is done separately, one decision for whether or not to challenge if you have 2 challenges remaining, another if you only have 1 challenge remaining.")
    st.markdown("Now, with these optimal challenge decisions as the ground truth, we train Machine Learning models to predict in which game situations players should challenge borderline pitches.")
    st.markdown("There are details on the optimization model and Machine Learning models used. However, below are the rules generated from the most basic ML model (decision tree with max depth of 3). Below are the rules that we could reasonably expect MLB players to consider when deciding whether or not to challenge a borderline pitch. If these rules are too specific, they can be generalized with: Only challenge early in the game if it is a great scoring chance, and always challenge late unless it is a very poor scoring chance.")
    st.markdown("If there are two challenges remaining:")
    st.markdown("- Always challenge if it is the 8th inning or later")
    st.markdown("- Only challenge in the 7th inning with less than 2 outs")
    st.markdown("- Only challenge in the 6th inning or earlier with runners in scoring position and less than 2 outs")
    st.markdown("If there is just one challenge remaining:")
    st.markdown("- Always challenge if it is the 9th inning")
    st.markdown("- Only challenge in the 8th inning if there are runners in scoring position")
    st.markdown("- Only challenge in the 7th inning or earlier with runners in scoring position and less than 2 outs")


with data_tab:
    st.subheader("Data Overview")
    st.write("For these models, we needed data on every borderline pitch from the 2026 MLB season (through August). To do this, I went on Statcast Search: https://baseballsavant.mlb.com/statcast_search. I used the following filters to create my dataset:")
    st.markdown("- Pitch Result: Ball or Called Strike")
    st.markdown("- Inning: 1 through 9 (omit extra innings)")
    st.markdown("- Attack Zones: 11-19 Shadow (because we only want to model borderline pitches)")
    st.markdown("- Game Date: 2026 season (through August) (had to separate this into two week periods in order to download everything)")
    st.markdown("- ABS Challenges: Is Challengeable Pitch")
    st.write("After downloading this data, I first made sure to revert any challenge decisions, as we want the dataset to reflect pre-challenge results. Then, for each pitch, I calculated the run value of the situation, and the counterfactual run value if the pitch were challenged and overturned. This provided a delta run value, or the value of successfully challenging the pitch.")
    st.write("Run value was calculated by taking into account 5 factors: baserunners, outs, count, batter quality, and pitcher quality.")
    st.write("Baserunners and outs were considered using the following Run Expectancy Matrix, which created the baseline run value of each situation: https://blogs.fangraphs.com/the-run-expectancy-matrix-reloaded-for-the-2020s/")
    run_ex_matrix = pd.read_csv("run_ex_matrix.csv")
    run_ex_height = (len(run_ex_matrix) * 35) + 37
    st.dataframe(run_ex_matrix,hide_index=True,height=run_ex_height)
    st.write("The impact of the count was taken into consideration using the wOBA ratio chart below, sourced from Fangraphs: https://blogs.fangraphs.com/the-count-is-king-even-after-accounting-for-batter-skill/. The wOBA ratio of the count in question was multiplied by the situational run value to scale the importance of the count.")
    woba_ratio = pd.read_csv("woba_ratio.csv")
    woba_ratio_height = (len(woba_ratio)*35)+37
    st.dataframe(woba_ratio,hide_index=True,height=woba_ratio_height)
    st.write("Batter and pitcher quality were determined by wOBA (batters) and wOBA against (pitchers), from Baseball Savant: https://baseballsavant.mlb.com/leaderboard/expected_statistics. Run value was multiplied by hitter quality (wOBA / 0.316, the league average wOBA) and pitcher quality (wOBA allowed / 0.316). Unqualified hitters and pitchers were assumed to be league average, leading to a multiplier of 1 (no impact).")
    st.write("Finally, the model considered static challenge success rates from the Baseball Savant ABS dashboard, as of 9/1/26: https://baseballsavant.mlb.com/abs. Hitters were assumed to have a challenge success rate of 0.49, and catchers 0.58.")
    st.write("While optimizing challenge usage, situational run value and challenge success probabilities were the only factors considered. When fitting ML models to predict whether or not to challenge, the following variables were used:")
    st.markdown("- Baserunners (binary variables on_1b, on_2b, on_3b, risp, and bases_loaded)")
    st.markdown("- Outs")
    st.markdown("- Inning")
    st.markdown("- Count (balls, strikes, and action_count, i.e. whether or not the call would lead to strike 3 or ball 4)")
    st.markdown("- Batter and pitcher quality (wOBA/wOBA allowed)")
    st.markdown("- Challenger (batter or pitcher)")

    

with optimization_tab:
    st.subheader("Framework for Optimizing ABS Challenge Usage")
    st.write("Decision: when should players use challenges on borderline pitches? In other words, what situations maximize the run value of challenges?")
    st.write("Model: For each game, globally optimize the use of ABS challenges, assuming we have complete hindsight, and know the run value and situation for every challengeable pitch. Only borderline pitches are considered, as players should always challenge obvious missed calls.")
    st.write("We also assume that challenge result probabilities are unknown, but constant: 0.49 for hitters, 0.58 for catchers, according to the Baseball Savant ABS dashboard, as of 9/1/26: https://baseballsavant.mlb.com/abs")
    st.write("For every pitch that one team can challenge in a game, we assign two binary decision variables: should you challenge this pitch result if you have both challenges remaining, and should you challenge this pitch result if you have only one challenge remaining.")
    st.subheader("Optimization with Backward Dynamic Programming (Bellman's Equations)")
    st.write("We can solve this optimization problem using backward dynamic programming, starting at the last pitch (which is an obvious decision, always challenge it if you can) and iterating forward, where the future value of retaining a challenge can be derived from previous pitch run value scenarios and challenge success probabilities. Mathematically, we can formulate Bellman's Equations as such:")
    st.markdown("$V(i,c)$ is the optimal challenge value from pitch $i$ onward, with $c$ challenges remaining")
    st.markdown("$\pi(i,c)$ is decision at pitch $i$ with $c$ challenges remaining that leads to the optimal challenge value")
    st.latex(r"r_i = \text{change in run value for a successful challenge of pitch } i")
    st.latex(r"p_i = \text{probability that a challenge of pitch } i \text{ is successful}")
    st.markdown("We must solve for decisions $\pi(i,c)$ that provide optimal value $V(i,c)$.")
    st.markdown("Future value of skipping a challenge is defined by:")
    st.latex(r"Q_{\text{skip}}(i,c) = V(i+1,c)")
    st.markdown("Future value of successfully challenging a pitch is defined by:")
    st.latex(r"Q_{\text{success}}(i,c) = V(i+1,c) + r_i")
    st.markdown("Future value of a failed challenge is defined by:")
    st.latex(r"Q_{\text{failure}}(i,c) = V(i+1,c-1)")
    st.markdown("Therefore, future value of a challenge is defined by:")
    st.latex(r"Q_{\text{challenge}}(i,c) = p_i Q_{\text{success}}(i,c) + (1-p_i)Q_{\text{failure}}(i,c)")
    st.markdown("Hence:")
    st.latex(r"V(i,c) = \max(Q_{\text{skip}}(i,c), Q_{\text{challenge}}(i,c)) \text{ for } c>0")
    st.markdown("""
    Firstly, we know that $\pi(N,c) = 1$ for $c>0$, meaning we always challenge the last pitch if we can. 
    This is because $V(N,c) = 0$, in other words future value after the last pitch is zero.
    Then, we can use this to calculate $V(N-1,c)$, which only depends on $V(N,c)=0$, $r_i$, and $p_i$, which are all known values.
    We continue recursively until we have calculated $V(i,c)$ and $\pi(i,c)$ for all $i$ and $c$.
    Now, we have optimized challenge usage iteratively, without the need for any optimization solver!
    """)

    st.subheader("Example of Backward Iteration to Optimize Challenge Decisions")
    st.write("Below is an example of how we iterate through pitches to make globally optimal challenge decisions. First, we start with only 1 challenge remaining:")
    st.markdown("Last pitch:")
    st.markdown("  - Success probability: 0.58 (catcher challenge)")
    st.markdown("  - Run value if challenged correctly = 0.07 (calculated based on situation)")
    st.markdown("  - Run value for skipping challenge = 0 (last pitch of the game")
    st.markdown("  - Run value for challenging = 0.04 = 0.58 * 0.07 (because this is the last pitch)")
    st.markdown("  - Optimal value = 0.04, with optimal decision to Challenge (since 0.04 > 0)")
    st.markdown("Second to last pitch:")
    st.markdown("  - Success probability: 0.58 (catcher challenge)")
    st.markdown("  - Run value if challenged correctly = 0.14 (calculated based on situation)")
    st.markdown("  - Run value for skipping challenge = 0.04 (from previous optimal value calculation above")
    st.markdown("  - Run value for challenging = 0.1 = 0.58(0.14+0.04) + (1-0.58)(0)")
    st.markdown("  - Optimal value = 0.1, with optimal decision to Challenge (since 0.1 > 0.04)")
    st.markdown("Third to last pitch:")
    st.markdown("  - Success probability: 0.49 (Hitter challenge)")
    st.markdown("  - Run value if challenged correctly = 0.1 (calculated based on situation)")
    st.markdown("  - Run value for skipping challenge = 0.1 (from previous optimal value calculation above")
    st.markdown("  - Run value for challenging = 0.098 = 0.49(0.1+0.1) + (1-0.49)(0)")
    st.markdown("  - Optimal value = 0.1, with optimal decision to NOT Challenge (since 0.098 < 0.1)")
    st.markdown("We continue iteratively, and then undergo the same process for 2 challenges remaining, until we have the results for two binary decision variables for each pitch, denoting whether we should challenge when there is 1 and 2 challenges remaining.")

def add_headers_to_cm(confusion_matrix):
    #add headers ("Predicted challenge, predicted no challenge, actual challenge, actual no challenge") to confusion matrix csv
    #st.dataframe ignores \n in header text, so render as HTML with <br> to force the line break
    confusion_matrix.columns = ["Predicted<br>challenge", "Predicted<br>no challenge"]
    confusion_matrix.index = ["Actual<br>challenge", "Actual<br>no challenge"]
    #make the correct predictions (top left and bottom right) light green, and the incorrect light red
    def color_cells(df):
        #diagonal cells (correct predictions) are light green, off-diagonal (incorrect) are light red
        styles = pd.DataFrame('background-color: lightcoral; color: black', index=df.index, columns=df.columns)
        for row_label, col_label in zip(df.index, df.columns):
            styles.loc[row_label, col_label] = 'background-color: lightgreen; color: black'
        return styles
    confusion_matrix = confusion_matrix.style.apply(color_cells, axis=None).format("{:.1f}%")
    #th cells are bold by default in pandas' HTML output; override to normal weight
    confusion_matrix = confusion_matrix.set_table_styles(
        [{"selector": "th", "props": [("font-weight", "normal")]}]
    )
    return confusion_matrix.to_html(escape=False)

with classifiers_tab:

    st.subheader("Machine Learning Classifiers")
    st.write("We train ML classifiers (using sklearn) on the results of the optimization model, in order to have way to predict whether or not it is worthwhile to use a challenge on a borderline pitch in any given situation. Four models are used:         ",
        "Decision Tree (depth of 3)",
        "Decision Tree (depth of 5)",
        "Logistic Regression",
        "Histogram Gradient Boosting.")
    st.write("For each model, there must be two separate rulesets: one when there are two challenges remaining, and another when there is only 1 challenge remaining. Realistically, the decision tree (depth of 3) is the only model that is quickly interpretable enough for players or coaches to use to make decisions during the game. The basic rules from the tree are outlined in the Methodology tab, but boil down to: Only challenge early in the game if it is a great scoring chance, and always challenge late unless it is a very poor scoring chance.")
    st.write("----")
    st.subheader("Decision Tree (depth of 3)")
    dt1, dt2 = st.columns(2)
    with dt1:
        st.write("2 Challenges remaining:")
        st.image("tree_2_left.png")
    with dt2:
        st.write("1 Challenge remaining:")
        st.image("tree_1_left.png")
    dt3, dt4 = st.columns(2)
    with dt3:
        st.write("Confusion Matrix with 2 challenges remaining:")
        cm_tree_2_left = pd.read_csv("cm_tree_2_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_tree_2_left), unsafe_allow_html=True)
    with dt4:
        st.write("Confusion Matrix with 1 challenge remaining:")
        cm_tree_1_left = pd.read_csv("cm_tree_1_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_tree_1_left), unsafe_allow_html=True)
    st.write("Takeaways")
    st.markdown("- Model is conservative, doing relatively well at predicting when not to challenge, but missing some challenge opportunities. This makes sense, considering it is an unbalanced dataset (more non-challenge situations than challenge situations).")
    st.markdown("- Model decisions boil down to being aggressive late in the game, and conservative early in the game, unless it is a really good scoring opportunity.")
    st.markdown("- Interestingly, model does not consider count, quality of the batter/pitcher, and is exactly the same no matter who is challenging (batter vs. catcher). This is somewhat to be expected, a shallow tree can only include so many variables.")
    st.write("----")
    st.subheader("Decision Tree (depth of 5)")
    dt15, dt25 = st.columns(2)
    with dt15:
        st.write("2 Challenges remaining:")
        st.image("tree_2_left_5.png")
    with dt25:
        st.write("1 Challenge remaining:")
        st.image("tree_1_left_5.png")
    dt5_1, dt5_2 = st.columns(2)
    with dt5_1:
        st.write("Confusion Matrix with 2 challenges remaining:")
        cm_tree_5_2_left = pd.read_csv("cm_tree_5_2_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_tree_5_2_left), unsafe_allow_html=True)
    with dt5_2:
        st.write("Confusion Matrix with 1 challenge remaining:")
        cm_tree_5_1_left = pd.read_csv("cm_tree_5_1_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_tree_5_1_left), unsafe_allow_html=True)
    st.write("Takeaways")
    st.markdown("- Model improves with max_depth of 5, but somewhat marginally, and is still conservative (better at predicting when not to challenge, as opposed to when to challenge).")
    st.markdown("- Model now includes count, but variables such as inning and baserunners are still more critical. Pitcher/batter quality, along with who is challenging, is still not included.")
    st.markdown("- Basic, overall takeaway remains the same, in that the model is conservative early (unless it is a great scoring chance), and aggressive late (unless it is a really poor scoring chance).")
    st.write("----")
    st.subheader("Logistic Regression")
    logistic_weights = pd.DataFrame({
        "Variable": feature_columns,
        "Weight (2 challenges remaining)": logistic_2_left.coef_[0],
        "Weight (1 challenge remaining)": logistic_1_left.coef_[0],
    })
    #sort by the largest-magnitude weight across either ruleset, most influential variables first
    logistic_weights["_sort_key"] = logistic_weights[
        ["Weight (2 challenges remaining)", "Weight (1 challenge remaining)"]
    ].abs().max(axis=1)
    logistic_weights = logistic_weights.sort_values("_sort_key", ascending=False).drop(columns="_sort_key")
    st.dataframe(
        logistic_weights.style.format({
            "Weight (2 challenges remaining)": "{:.3f}",
            "Weight (1 challenge remaining)": "{:.3f}",
        }),
        hide_index=True,
        height = (len(logistic_weights) * 35 ) + 37
    )
    log1, log2 = st.columns(2)
    with log1:
        st.write("Confusion Matrix with 2 challenges remaining:")
        cm_logistic_2_left = pd.read_csv("cm_logistic_2_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_logistic_2_left), unsafe_allow_html=True)
    with log2:
        st.write("Confusion Matrix with 1 challenge remaining:")
        cm_logistic_1_left = pd.read_csv("cm_logistic_1_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_logistic_1_left), unsafe_allow_html=True)
    st.write("Takeaways")
    st.markdown("- Model performs similarly to Decision Trees, in terms of the confusion matrix and its overall conservative decision making.")
    st.markdown("- Unlike the Decision Trees, this model strongly considers the strength of the batter and the pitcher. The inning is still important, despite a smaller weight value, because it is one of the only variables that is not binary.")
    st.markdown("- On first glance, the negative value for bases loaded, and the small coefficient for risp seem counterintuitive. However, these make sense because of their collinearity with the large positive coefficient variables on_3b, on_2b, and on_1b.")
    st.write("----")
    st.subheader("Histogram Gradient Boosting")
    shap_weights = pd.DataFrame({
        "Variable": feature_columns,
        "Importance (2 challenges remaining)": hist_gb_2_left_mean_importance,
        "Importance (1 challenge remaining)": hist_gb_1_left_mean_importance,
    })
    shap_weights = shap_weights.sort_values(
        "Importance (2 challenges remaining)", ascending=False
    )
    st.dataframe(
        shap_weights.style.format({
            "Importance (2 challenges remaining)": "{:.3f}",
            "Importance (1 challenge remaining)": "{:.3f}",
        }),
        hide_index=True,
        height = (len(shap_weights) * 35 ) + 37
    )
    hgb1, hgb2 = st.columns(2)
    with hgb1:
        st.write("Confusion Matrix with 2 challenges remaining:")
        cm_hist_gb_2_left = pd.read_csv("cm_hist_gb_2_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_hist_gb_2_left), unsafe_allow_html=True)
    with hgb2:
        st.write("Confusion Matrix with 1 challenge remaining:")
        cm_hist_gb_1_left = pd.read_csv("cm_hist_gb_1_left.csv", header=None)
        st.markdown(add_headers_to_cm(cm_hist_gb_1_left), unsafe_allow_html=True)
    st.write("Takeaways")
    st.markdown("- This is the least interpretable model, and does seem to perform the best overall. It is less conservative than the others, although it still skews conservative in challenge decision making.")
    st.markdown("- Importance values calculated using Permutation Importance from sklearn. Using these values, we see similar patterns as in the Decision Trees, with inning as the most important feature, and batter/pitcher wOBA not being very impactful.")
    st.markdown("- Similarly to the Logistic Regression model, collinearity in features must be considered when analyzing importance values, specifically for variables such as action_count and bases_loaded, which are collinear with other count and baserunner variables.")
with limitations_tab:
    st.subheader("Limitations")
    st.markdown("- Model decisions (challenge or no challenge) are of somewhat limited value, due to the relatively poor performance of the models (especially the interpretable ones)")
    st.markdown("- The models are difficult to use in real game situations, even the decision trees are not the most practical to use. The most value of the models could come from the general takeaways/strategy, not the specific rules.")
    st.markdown("- Models skew conservatively, because they are built on a skewed dataset (more no-challenge situations than challenge situations). To fix this, future work could include upsampling, or adjusting the reward function that trains the ML models to reward successfully challenging.")
    st.markdown("- Because the optimization is done only considering borderline pitches, the value of keeping a challenge for obvious, non-borderline missed calls is not included. This is one reason why the conservative nature of the models may be valuable in practice.")
    st.markdown("- The models do not take into consideration the individual player challenge success probabilities, as some players have had more success challenging than others. This data is currently not included in the model.")

st.write("---")
st.write("Project created by Malcolm Gaynor. Please don't hesitate to reach out with any questions or comments: malcolm.t.gaynor@gmail.com")
