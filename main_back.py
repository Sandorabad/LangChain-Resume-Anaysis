import os
import random
import PyPDF2
import pandas as pd
from langchain import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI



# ---------------------------------------------------------------------------
# 0. PDF Resume Extraction and Candidate Loading
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def load_candidates_from_folder(resume_folder): # TO START
    """
    Loads all PDF resumes from a folder.
    The candidate name is inferred from the PDF file name.
    Each candidate is stored as a dictionary with an 'Explanations' list to accumulate round explanations.
    """
    candidates = []
    for file in os.listdir(resume_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(resume_folder, file)
            candidate = {
                "File Name": file.split(".")[0],
                "Resume Path": pdf_path,
                "Resume Text": extract_text_from_pdf(pdf_path),
                "Explanations": []
            }
            candidates.append(candidate)
    return candidates

def load_selected_candidates(resume_folder, selected_resumes): # FOR THE 2ND AND BEYOND ROUNDS
    """
    Loads only the selected PDF resumes from a folder.
    The candidate name is inferred from the PDF file name.
    Each candidate is stored as a dictionary with an 'Explanations' list to accumulate round explanations.
    """
    candidates = []
    selected_resumes_set = {os.path.basename(resume) for resume in selected_resumes}

    for file in os.listdir(resume_folder):
        if file in selected_resumes_set and file.lower().endswith(".pdf"):
            pdf_path = os.path.join(resume_folder, file)
            candidate = {
                "File Name": file.split(".")[0],
                "Resume Path": pdf_path,
                "Resume Text": extract_text_from_pdf(pdf_path),
                "Explanations": []
            }
            candidates.append(candidate)

    return candidates

# -------------------------------------------------------------------------------
# 1. OpenAI Model Setup # REMEMBER TO SET YOU API KEY AS AN ENVIROMENT VARIABLE
# -------------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# 2. Define the Doble Comparison Chain
# ---------------------------------------------------------------------------
doble_prompt_template = """
Round {round_number}: Compare the two resumes below based on the following job description.
Decide which candidate is a better fit and provide explanations. Be carefull with overqualified candidates.

Job Description:
{job_description}

Resume 1 (File Name: {candidate1_filename}):
{resume_text1}

Resume 2 (File Name: {candidate2_filename}):
{resume_text2}

Please respond in the following structured format:
Winner: <Candidate Name>
Winner Resume: <Resume Path>
Winner Explanation: <Explanation Round {round_number}: why the winner was selected>
Loser: <Candidate Name>
Losser Resume: <Resume Path>
Loser Explanation: <Explanation Round {round_number}: why the loser was not selected>
"""

doble_prompt = PromptTemplate(
    template=doble_prompt_template,
    input_variables=["round_number", "job_description", "candidate1_name", "candidate1_filename", "resume_text1", "candidate2_name", "candidate2_filename", "resume_text2"]
)
doble_chain = LLMChain(llm=llm, prompt=doble_prompt)


# ---------------------------------------------------------------------------
# 3. Define the Triple Comparison Chain
# ---------------------------------------------------------------------------
triple_prompt_template = """
Round {round_number}: Compare the three resumes below based on the following job description.
Decide on a ranking for these candidates and provide explanations. Be carefull with overqualified candidates.

Job Description:
{job_description}

Resume 1 (File Name: {candidate1_filename}):
{resume_text1}

Resume 2 (File Name: {candidate2_filename}):
{resume_text2}

Resume 3 (File Name: {candidate3_filename}):
{resume_text3}

Please respond in the following structured format:
Winner: <Candidate Name>
Winner Resume: <Resume Path>
Winner Explanation: <Explanation Round {round_number}: why the winner was selected>
Runner-up: <Candidate Name>
Runner-up Resume: <Resume Path>
Runner-up Explanation: <Explanation Round {round_number}: why the runner-up was ranked second>
Third: <Candidate Name>
Third Resume: <Resume Path>
Third Explanation: <Explanation Round {round_number}: why the candidate ranked third was not selected>
"""

triple_prompt = PromptTemplate(
    template=triple_prompt_template,
    input_variables=["round_number", "job_description", "candidate1_name", "candidate1_filename" "resume_text1", "candidate2_name", "candidate2_filename", "resume_text2", "candidate3_name", "candidate3_filename", "resume_text3"]
)
triple_chain = LLMChain(llm=llm, prompt=triple_prompt)


# ---------------------------------------------------------------------------
# 4. Tournament Round Function
# ---------------------------------------------------------------------------
def tournament_round(candidates, job_description, round_number):
    """
    Runs one round of the tournament.
    Returns:
      - A list of winners advancing to the next round.
      - A list of match details for this round.
    """
    next_round_resumes = []
    match_details = []
    n = len(candidates)
    
    # Reserve the last 3 candidates for a triple match if the total is odd.
    pair_end = n - 3 if (n % 2 == 1 and n >= 3) else n

    # Process pairwise matches.
    i = 0
    while i < pair_end:
        candidate1 = candidates[i]
        candidate2 = candidates[i+1]
        
        result = doble_chain.run(
            round_number = round_number,
            job_description = job_description,
            candidate1_filename = candidate1["Resume Path"],
            resume_text1 = candidate1["Resume Text"],
            candidate2_filename = candidate2["Resume Path"],
            resume_text2 = candidate2["Resume Text"],
            )
        
        
        match_result = {}
        for line in result.strip().splitlines():
            line = line.strip().replace("**", "")
            # WINNER
            if line.startswith("Winner Resume:"):
                match_result["Winner_Resume"] = line.split("Winner Resume:")[-1].strip()
            elif line.startswith("Winner Explanation:"):
                match_result["Winner_Explanation"] = line.split("Winner Explanation:")[-1].strip()
            elif line.startswith("Winner:"):
                match_result["Winner_Name"] = line.split("Winner:")[-1].strip()
            # LOSER
            elif line.startswith("Losser Resume:"):
                match_result["Losser_Resume"] = line.split("Loser Resume:")[-1].strip()
            elif line.startswith("Loser:"):
                match_result["Loser_Name"] = line.split("Loser:")[-1].strip()
            elif line.startswith("Loser Explanation:"):
                match_result["Loser_Explanation"] = line.split("Loser Explanation:")[-1].strip()
              
        # GET THE NEXT TWO FOR MORTAL KOMBAT
        i += 2
        # Append the Winner
        next_round_resumes.append(match_result["Winner_Resume"])
        # Append The Match_details
        match_details.append(match_result)

    # Process triple match for the last three candidates if needed.
    if n - pair_end == 3:
        candidate1 = candidates[pair_end]
        candidate2 = candidates[pair_end+1]
        candidate3 = candidates[pair_end+2]
        
        result = triple_chain.run(
            round_number = round_number,
            job_description = job_description,
            candidate1_filename = candidate1["Resume Path"],
            resume_text1 = candidate1["Resume Text"],
            candidate2_filename = candidate2["Resume Path"],
            resume_text2 = candidate2["Resume Text"],
            candidate3_filename = candidate3["Resume Path"],
            resume_text3 = candidate3["Resume Text"]
        )
        
        match_result = {}
        for line in result.strip().splitlines():
            line = line.strip().replace("**", "")
            # WINNER
            if line.startswith("Winner Resume:"):
                match_result["Winner_Resume"] = line.split("Winner Resume:")[-1].strip()
            elif line.startswith("Winner Explanation:"):
                match_result["Winner_Explanation"] = line.split("Winner Explanation:")[-1].strip()
            elif line.startswith("Winner:"):
                match_result["Winner_Name"] = line.split("Winner:")[-1].strip()
            # RUNNER-UP
            elif line.startswith("Runner-up Resume:"):
                match_result["Runner_Up_Resume"] = line.split("Runner-up Resume:")[-1].strip()
            elif line.startswith("Runner-up:"):
                match_result["Runner_Up_Name"] = line.split("Runner-up:")[-1].strip()
            elif line.startswith("Runner-up Explanation:"):
                match_result["Runner_Up_Explanation"] = line.split("Runner-up Explanation:")[-1].strip()
            # THIRD PLACE
            elif line.startswith("Third Resume:"):
                match_result["Third_Resume"] = line.split("Third Resume:")[-1].strip()
            elif line.startswith("Third:"):
                match_result["Third_Name"] = line.split("Third:")[-1].strip()
            elif line.startswith("Third Explanation:"):
                match_result["Third_Explanation"] = line.split("Third Explanation:")[-1].strip()
        # Append the Winner
        next_round_resumes.append(match_result["Winner_Resume"])
        # Append The Match_details
        match_details.append(match_result)

    print(f"Completed Round {round_number}: {len(next_round_resumes)} Candidates Advancing.")
    return next_round_resumes, match_details


# ---------------------------------------------------------------------------
# 6. GENERATE EXCEL FILE
# ---------------------------------------------------------------------------
def generate_excel_results(tournament_results):
    data = []
    for round_num, matches in enumerate(tournament_results, start=1):
        for match_num, match in enumerate(matches, start=1):
            row = {
                "Round": round_num,
                "Match": match_num,
                "Winner": match.get("Winner_Name", None),
                "Winner Resume": match.get("Winner_Resume", None),
                "Winner Explanation": match.get("Winner_Explanation", None),
                "Loser": match.get("Loser_Name", None),
                "Loser Resume": match.get("Losser_Resume", None),
                "Loser Explanation": match.get("Loser_Explanation", None),
                "Runner-Up": match.get("Runner_Up_Name", None),
                "Runner-Up Resume": match.get("Runner_Up_Resume", None),
                "Runner-Up Explanation": match.get("Runner_Up_Explanation", None),
                "Third": match.get("Third_Name", None),
                "Third Resume": match.get("Third_Resume", None),
                "Third Explanation": match.get("Third_Explanation", None)
            }
            data.append(row)
    # Create a DataFrame from the flattened data.
    df = pd.DataFrame(data)
    # Export the DataFrame to an Excel file.
    df.to_excel("tournament_results.xlsx", index=False)
    print("Tournament results exported to tournament_results.xlsx")

# ---------------------------------------------------------------------------
# 6. MORTAL KOMBAT
# ---------------------------------------------------------------------------
def run_matches(resume_folder, job_description):
    # We want to save the Tournament history:
    tournament_history = []

    # Initial Candidate list
    candidates = load_candidates_from_folder(resume_folder)
    print("Starting Selection / Tournament")
    print(f"N° of Candidates: {len(candidates)}")

    # Round 1
    next_round_resumes, match_details = tournament_round(candidates = candidates,
                                                     job_description = job_description,
                                                     round_number = 1)
    tournament_history.append(match_details) # Save to memory(card)

    # MORTAL KOMBAT! (tutututututututu Choose your Destiny)
    while len(next_round_resumes) != 1:
        # Load Smaller Batch of Candidates:
        round = 2
        candidates = load_selected_candidates(resume_folder = resume_folder, selected_resumes = next_round_resumes)
        
        # Run the Tournament
        next_round_resumes, match_details = tournament_round(candidates = candidates,
                                                     job_description = job_description,
                                                     round_number = round)
        tournament_history.append(match_details) # Save to memory(card)
        round =+ 1
    
    print(f"Winner: {next_round_resumes}")
    generate_excel_results(tournament_results = tournament_history)
    return print("END")
