import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Introduction",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

# Title of the page
st.image('./images/Bc1.jpg')
st.title("Breast Cancer Overview")


# Overview section
st.header("Overview")
st.write("""
Breast cancer occurs when breast cells mutate and become cancerous cells that multiply and form tumors. It typically affects women and people assigned female at birth (AFAB) age 50 and older, but it can also affect men and people assigned male at birth (AMAB), as well as younger women. 
Healthcare providers may treat breast cancer with surgery to remove tumors or treatment to kill cancerous cells.
""")

st.subheader("What is breast cancer?")
st.image('./images/2.jpg')
st.write("""
Breast cancer is one of the most common cancers that affects women and people AFAB. It happens when cancerous cells in the breasts multiply and become tumors. About 80% of breast cancer cases are invasive, meaning a tumor may spread from your breast to other areas of your body.
Breast cancer typically affects women age 50 and older, but it can also affect women and people AFAB who are younger than 50. Men and people assigned male at birth (AMAB) may also develop breast cancer.
""")

st.subheader("Breast cancer types")
st.write("""
Healthcare providers determine cancer types and subtypes so they can tailor treatment to be as effective as possible with the fewest possible side effects. Common types of breast cancer include:

- Invasive (infiltrating) ductal carcinoma (IDC): This cancer starts in your milk ducts and spreads to nearby breast tissue. It’s the most common type of breast cancer in the United States.
- Lobular breast cancer: This cancer starts in the milk-producing glands (lobules) and often spreads to nearby breast tissue. It’s the second most common breast cancer in the United States.
- Ductal carcinoma in situ (DCIS): Like IDC, this cancer starts in your milk ducts but doesn’t spread beyond the ducts.

Other types include triple-negative breast cancer (TNBC), inflammatory breast cancer (IBC), and Paget’s disease of the breast.
""")

st.subheader("Breast cancer subtypes")
st.write("""
Breast cancer subtypes are classified by receptor cell status:

- ER-positive (ER+): Has estrogen receptors.
- PR-positive (PR+): Has progesterone receptors.
- HR-positive (HR+): Has both estrogen and progesterone receptors.
- HR-negative (HR-): Lacks these hormone receptors.
- HER2-positive (HER2+): Has higher than normal HER2 protein levels. About 15% to 20% of all breast cancers are HER2-positive.
""")

# Symptoms and Causes section
st.header("Symptoms and Causes")
st.image('./images/3.jpg')
st.subheader("What are breast cancer symptoms?")
st.write("""
The condition can affect your breasts in different ways. Symptoms include:

- A change in the size, shape, or contour of your breast.
- A lump or thickening near your breast or in your underarm.
- Skin changes (dimpling, puckering, or inflamed skin).
- Discharge from your nipple, which could be clear or blood-stained.
""")

st.subheader("What causes breast cancer?")
st.write("""
Breast cancer occurs when breast cells mutate, but the exact cause is unknown. Risk factors include:

- Age: Being 55 or older.
- Sex: Women and people AFAB are at higher risk.
- Family history: A close family member with breast cancer increases your risk.
- Genetics: Inherited mutations, especially BRCA1 and BRCA2 genes.
- Smoking, alcohol use, obesity, and radiation exposure.
""")

# Diagnosis and Tests section
st.header("Diagnosis and Tests")
st.image('./images/4.gif')
st.write("""
To diagnose breast cancer, healthcare providers may perform:

- Breast ultrasound or MRI scans.
- Breast biopsy.
- Immunohistochemistry tests to check for hormone receptors.
- Genetic tests to identify mutations.

Breast cancer is classified into different stages:

- Stage 0: Noninvasive, meaning it hasn’t spread.
- Stage I-IV: Based on tumor size, spread to lymph nodes, and metastasis.
""")

# Management and Treatment section
st.header("Management and Treatment")
st.write("""
Surgery is often the primary treatment, such as a mastectomy or lumpectomy. Other treatments include:

- Chemotherapy
- Radiation therapy
- Immunotherapy
- Hormone therapy (e.g., selective estrogen receptor modulators)
- Targeted therapy

Side effects of treatment vary but may include nausea, fatigue, gastrointestinal issues, and infection risk from surgery.
""")

# Prevention section
st.header("Prevention")
st.write("""
While breast cancer can’t always be prevented, these steps can reduce risk:

- Maintain a healthy weight.
- Eat a balanced diet rich in fruits and vegetables.
- Stay physically active.
- Limit alcohol consumption.
- Regular screenings, including mammograms and self-exams.
""")

# Outlook and Prognosis section
st.header("Outlook / Prognosis")
st.write("""
Survival rates depend on the cancer’s stage and type. Generally, the five-year survival rate for localized cancer is 99%. Early detection and treatment improve outcomes significantly. However, metastatic breast cancer, which spreads to other organs, has a five-year survival rate of about 30%.
""")

# Living with Breast Cancer
st.header("Living With Breast Cancer")
st.write("""
Managing breast cancer can be challenging. It’s important to get enough rest, eat a balanced diet, manage stress, and seek support through survivorship programs. 

Regular check-ins with healthcare providers can help monitor for any new symptoms or complications.
""")

# Additional questions section
st.header("Additional Common Questions")
st.write("""
- How long can you have breast cancer without knowing? You can have breast cancer for years before noticing symptoms.
- Can men get breast cancer? Yes, although it's rare, men and people AMAB can develop breast cancer.
""")
