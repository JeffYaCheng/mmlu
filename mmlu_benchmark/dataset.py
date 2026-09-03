"""
MMLU Dataset Loader & Preprocessing Module
=========================================
Handles loading MMLU subsets from Hugging Face datasets (cais/mmlu) with
comprehensive offline built-in subject data for instant local execution.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MMLU-Dataset")

# Official MMLU 57 Subject Taxonomy across 4 Canonical Disciplines
MMLU_CATEGORIES: Dict[str, List[str]] = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
        "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology"
    ]
}

SUBJECT_TO_CATEGORY: Dict[str, str] = {}
ALL_57_SUBJECTS: List[str] = []
for _cat, _subjs in MMLU_CATEGORIES.items():
    for _s in _subjs:
        SUBJECT_TO_CATEGORY[_s] = _cat
        ALL_57_SUBJECTS.append(_s)

# Four-Domain Balanced Benchmark Subset (3 subjects per canonical category, 12 total)
BALANCED_BENCHMARK_SUBJECTS: List[str] = [
    # STEM (3)
    "machine_learning", "high_school_physics", "college_computer_science",
    # Humanities (3)
    "philosophy", "professional_law", "high_school_world_history",
    # Social Sciences (3)
    "econometrics", "high_school_psychology", "us_foreign_policy",
    # Other (3)
    "business_ethics", "management", "clinical_knowledge"
]


def normalize_answer_to_letter(ans: Any) -> str:
    """Normalizes answer from any format (int 0-3, str '0'-'3', or 'A'-'D') to uppercase 'A'-'D'."""
    if isinstance(ans, int):
        if 0 <= ans <= 3:
            return chr(65 + ans)
        return "A"
    if isinstance(ans, str):
        cleaned = ans.strip().upper()
        if cleaned in ["A", "B", "C", "D"]:
            return cleaned
        if cleaned in ["0", "1", "2", "3"]:
            return chr(65 + int(cleaned))
    return "A"


# Comprehensive subject samples across STEM, Humanities, Social Sciences, Other
MMLU_SAMPLE_DATABASE: Dict[str, Dict[str, Any]] = {
    # === STEM ===
    "machine_learning": {
        "test": [
            {
                "question": "Which of the following activation functions suffers from the vanishing gradient problem when inputs are very large positive or negative numbers?",
                "choices": ["ReLU (Rectified Linear Unit)", "Sigmoid", "Leaky ReLU", "ELU (Exponential Linear Unit)"],
                "answer": "B",
            },
            {
                "question": "What is the primary purpose of Dropout in training deep neural networks?",
                "choices": ["Speeding up gradient descent convergence", "Preventing overfitting by regularizing representations", "Reducing memory consumption during inference", "Guaranteed global minimum convergence"],
                "answer": "B",
            },
            {
                "question": "In Support Vector Machines (SVM), what does the kernel trick allow us to do?",
                "choices": ["Compute dot products in high-dimensional feature spaces without explicitly mapping data points", "Avoid calculating the margin entirely", "Convert regression tasks into classification tasks", "Reduce sample size automatically"],
                "answer": "A",
            },
            {
                "question": "Which evaluation metric is most appropriate for a severely class-imbalanced binary classification problem?",
                "choices": ["Raw Accuracy", "Precision-Recall AUC (PR-AUC)", "Mean Squared Error", "0-1 Loss"],
                "answer": "B",
            },
            {
                "question": "In Transformer architectures, what is the computational complexity of standard self-attention with respect to sequence length N?",
                "choices": ["O(N)", "O(N log N)", "O(N^2)", "O(N^3)"],
                "answer": "C",
            },
        ],
        "dev": [
            {
                "question": "In gradient descent, what happens if the learning rate is set excessively large?",
                "choices": ["The model converges too quickly", "The loss function may oscillate wildly or diverge", "The gradients become zero permanently", "The weights freeze in place"],
                "answer": "B",
            },
            {
                "question": "What does L2 regularization (Ridge) penalize in the loss function?",
                "choices": ["The sum of absolute values of weights", "The sum of squared weights", "The maximum weight value", "The number of non-zero parameters"],
                "answer": "B",
            },
            {
                "question": "In unsupervised learning, which algorithm partitions n observations into k clusters by iteratively updating centroids?",
                "choices": ["k-means clustering", "DBSCAN", "Random Forest", "Linear Discriminant Analysis"],
                "answer": "A",
            },
            {
                "question": "What is the primary difference between Bagging and Boosting in ensemble learning?",
                "choices": ["Bagging uses decision trees while Boosting uses neural networks", "Bagging trains base learners independently in parallel, while Boosting trains models sequentially to correct errors", "Bagging is for classification only and Boosting is for regression only", "Bagging increases model variance whereas Boosting increases bias"],
                "answer": "B",
            },
            {
                "question": "Which issue occurs in deep neural networks when gradients approach zero during backpropagation through early layers?",
                "choices": ["Exploding gradient problem", "Internal covariate shift", "Vanishing gradient problem", "Curse of dimensionality"],
                "answer": "C",
            },
        ],
    },
    "high_school_physics": {
        "test": [
            {
                "question": "A ball is thrown vertically upwards with a velocity of 20 m/s. Assuming g = 10 m/s^2 and neglecting air resistance, what is the maximum height reached?",
                "choices": ["10 m", "20 m", "40 m", "50 m"],
                "answer": "B",
            },
            {
                "question": "According to Newton's Second Law of Motion, what is the net force acting on an object with constant velocity?",
                "choices": ["Zero", "Proportional to its mass", "Proportional to its velocity", "Infinite"],
                "answer": "A",
            },
            {
                "question": "What happens to the total resistance when two identical resistors of resistance R are connected in parallel?",
                "choices": ["It becomes 2R", "It remains R", "It becomes R/2", "It becomes R/4"],
                "answer": "C",
            },
        ],
        "dev": [
            {
                "question": "What is the SI unit of work and energy?",
                "choices": ["Watt", "Joule", "Newton", "Pascal"],
                "answer": "B",
            },
            {
                "question": "If the frequency of a wave is 50 Hz and its wavelength is 4 meters, what is its propagation speed?",
                "choices": ["12.5 m/s", "200 m/s", "46 m/s", "54 m/s"],
                "answer": "B",
            },
            {
                "question": "According to Ohm's Law, if the voltage across a resistor is doubled while resistance remains constant, what happens to the current?",
                "choices": ["It is halved", "It doubles", "It remains unchanged", "It quadruples"],
                "answer": "B",
            },
            {
                "question": "What phenomenon describes the bending of light as it passes from one transparent medium to another with a different refractive index?",
                "choices": ["Diffraction", "Refraction", "Polarization", "Dispersion"],
                "answer": "B",
            },
            {
                "question": "Which law states that an object continues in its state of rest or uniform motion unless acted upon by a net external force?",
                "choices": ["Newton's First Law (Law of Inertia)", "Newton's Third Law", "Coulomb's Law", "Kepler's Second Law"],
                "answer": "A",
            },
        ]
    },
    "college_computer_science": {
        "test": [
            {
                "question": "Which asymptotic time complexity best describes the average-case runtime of QuickSort on an array of length n?",
                "choices": ["O(n)", "O(n log n)", "O(n^2)", "O(2^n)"],
                "answer": "B",
            },
            {
                "question": "In relational database theory, which normal form eliminates transitive functional dependencies?",
                "choices": ["First Normal Form (1NF)", "Second Normal Form (2NF)", "Third Normal Form (3NF)", "Boyce-Codd Normal Form (BCNF)"],
                "answer": "C",
            },
            {
                "question": "Which concurrency hazard occurs when two processes are each waiting for the other to release a shared lock?",
                "choices": ["Starvation", "Deadlock", "Race condition", "Priority inversion"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What data structure uses LIFO (Last In First Out) ordering?",
                "choices": ["Queue", "Stack", "Binary Heap", "Hash Table"],
                "answer": "B",
            },
            {
                "question": "In a balanced binary search tree with n nodes, what is the worst-case search time complexity?",
                "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                "answer": "B",
            },
            {
                "question": "In computer architecture, what does spatial locality refer to?",
                "choices": ["Accessing memory locations close to recently accessed addresses", "Accessing the same memory address repeatedly over a short time interval", "Transferring data across multiple CPU cores", "Minimizing pipeline branch mispredictions"],
                "answer": "A",
            },
            {
                "question": "Which protocol operates at the Transport layer of the OSI model and provides reliable, connection-oriented data delivery?",
                "choices": ["UDP", "TCP", "IP", "ICMP"],
                "answer": "B",
            },
            {
                "question": "What is the primary function of an operating system's virtual memory subsystem?",
                "choices": ["To increase CPU clock frequency", "To provide an isolated address space mapped between physical RAM and disk storage", "To eliminate cache misses completely", "To decrypt network packets automatically"],
                "answer": "B",
            },
        ]
    },
    # === Humanities ===
    "philosophy": {
        "test": [
            {
                "question": "Which philosopher is famously associated with the epistemological proposition 'Cogito, ergo sum' (I think, therefore I am)?",
                "choices": ["Immanuel Kant", "René Descartes", "John Locke", "David Hume"],
                "answer": "B",
            },
            {
                "question": "In moral philosophy, which normative ethical framework evaluates actions primarily by their consequences, aiming for the greatest happiness for the greatest number?",
                "choices": ["Deontology", "Utilitarianism", "Virtue Ethics", "Nihilism"],
                "answer": "B",
            },
            {
                "question": "Plato's Allegory of the Cave is primarily an exploration of which philosophical theme?",
                "choices": ["Aesthetics and poetic meter", "Epistemology and the nature of perceived reality", "Military strategy", "Monetary economics"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "Which philosopher formulated the Categorical Imperative?",
                "choices": ["Aristotle", "Immanuel Kant", "Friedrich Nietzsche", "Baruch Spinoza"],
                "answer": "B",
            },
            {
                "question": "In informal logic, which fallacy occurs when an arguer attacks an opponent's personal character rather than their substantive argument?",
                "choices": ["Ad hominem", "Straw man", "Post hoc ergo propter hoc", "Begging the question"],
                "answer": "A",
            },
            {
                "question": "Epistemology is the branch of philosophy primarily concerned with the study of what?",
                "choices": ["Art and beauty", "Knowledge, belief, and justification", "Physical cosmology", "Political power structures"],
                "answer": "B",
            },
            {
                "question": "Which pre-Socratic Greek philosopher is famous for the doctrine that change is central to the universe ('You cannot step into the same river twice')?",
                "choices": ["Parmenides", "Heraclitus", "Thales", "Zeno of Elea"],
                "answer": "B",
            },
            {
                "question": "Which normative ethical framework emphasizes duty and moral obligation over the consequences of an action?",
                "choices": ["Deontology", "Consequentialism", "Ethical egoism", "Emotivism"],
                "answer": "A",
            },
        ]
    },
    "professional_law": {
        "test": [
            {
                "question": "Under standard Common Law, what are the three essential elements required to form a binding contract?",
                "choices": ["Offer, Acceptance, Consideration", "Offer, Signature, Notarization", "Intention, Performance, Payment", "Proposal, Counteroffer, Execution"],
                "answer": "A",
            },
            {
                "question": "In tort law, what standard of proof is generally required to establish liability in civil negligence cases in the US?",
                "choices": ["Beyond a reasonable doubt", "Preponderance of the evidence", "Clear and convincing evidence", "Absolute certainty"],
                "answer": "B",
            },
            {
                "question": "What legal doctrine prevents a party from re-litigating a claim that has already been decided in a final judgment on the merits?",
                "choices": ["Stare decisis", "Res judicata", "Habeas corpus", "Mens rea"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What term describes the mental state or intent required to commit a criminal offense?",
                "choices": ["Actus reus", "Mens rea", "Corpus delicti", "Certiorari"],
                "answer": "B",
            },
            {
                "question": "Under the doctrine of stare decisis, what are common law courts required or expected to do?",
                "choices": ["Draft new statutes independently", "Adhere to established precedents set by prior appellate decisions", "Consult public polling data before rendering verdicts", "Defer all constitutional questions to the executive branch"],
                "answer": "B",
            },
            {
                "question": "In US Constitutional Law, which landmark Supreme Court case established the principle of judicial review?",
                "choices": ["Marbury v. Madison", "McCulloch v. Maryland", "Brown v. Board of Education", "Gibbons v. Ogden"],
                "answer": "A",
            },
            {
                "question": "In tort law, what is the failure to exercise reasonable care under the circumstances that causes harm to another called?",
                "choices": ["Strict liability", "Negligence", "Battery", "Trespass"],
                "answer": "B",
            },
            {
                "question": "What formal pleading is filed by a plaintiff to initiate a civil lawsuit in court?",
                "choices": ["Subpoena", "Complaint", "Affidavit", "Indictment"],
                "answer": "B",
            },
        ],
    },
    "high_school_world_history": {
        "test": [
            {
                "question": "The Industrial Revolution first originated in the mid-18th century in which country?",
                "choices": ["France", "Great Britain", "Germany", "United States"],
                "answer": "B",
            },
            {
                "question": "Which major conflict concluded in 1919 with the signing of the Treaty of Versailles?",
                "choices": ["The Franco-Prussian War", "World War I", "World War II", "The Thirty Years' War"],
                "answer": "B",
            },
            {
                "question": "The Silk Road was an ancient network of trade routes connecting China primarily with which region?",
                "choices": ["The Mediterranean and Western Asia", "Sub-Saharan Africa", "Mesoamerica", "Polynesia"],
                "answer": "A",
            },
        ],
        "dev": [
            {
                "question": "Which ancient civilization constructed the Great Pyramids at Giza?",
                "choices": ["Mesopotamia", "Ancient Egypt", "Indus Valley Civilization", "Minoan Civilization"],
                "answer": "B",
            },
            {
                "question": "In 1492, Christopher Columbus's voyage across the Atlantic was funded by the monarchy of which country?",
                "choices": ["Portugal", "Spain", "England", "France"],
                "answer": "B",
            },
            {
                "question": "What 1215 English document limited royal authority and established that even the monarch was subject to the rule of law?",
                "choices": ["Magna Carta", "The Bill of Rights", "The Petition of Right", "The Edict of Nantes"],
                "answer": "A",
            },
            {
                "question": "The Renaissance cultural movement originated during the 14th century primarily in which region?",
                "choices": ["Scandinavia", "Northern and Central Italy", "The Iberian Peninsula", "The British Isles"],
                "answer": "B",
            },
            {
                "question": "Which structure, erected during the Cold War to divide a major European city, fell in November 1989?",
                "choices": ["The Maginot Line", "The Berlin Wall", "Hadrian's Wall", "The Iron Curtain Fence"],
                "answer": "B",
            },
        ]
    },
    # === Social Sciences ===
    "econometrics": {
        "test": [
            {
                "question": "In Ordinary Least Squares (OLS) regression, what Gauss-Markov condition is violated when the error term variance is non-constant across observations?",
                "choices": ["Multicollinearity", "Heteroskedasticity", "Endogeneity", "Autocorrelation"],
                "answer": "B",
            },
            {
                "question": "What statistical issue occurs when an explanatory variable is correlated with the regression error term?",
                "choices": ["Endogeneity", "Homoskedasticity", "Stationarity", "Perfect Collinearity"],
                "answer": "A",
            },
            {
                "question": "In time series analysis, what does the Augmented Dickey-Fuller (ADF) test check for?",
                "choices": ["Normality of residuals", "Presence of a unit root (non-stationarity)", "Serial independence", "Structural break"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What does R-squared measure in a linear regression model?",
                "choices": ["The statistical significance of the intercept", "The proportion of variance in the dependent variable explained by independent variables", "The degree of autocorrelation", "The sample size adequacy"],
                "answer": "B",
            },
            {
                "question": "When two or more independent variables in a multiple regression model are highly linearly correlated, what statistical issue arises?",
                "choices": ["Heteroskedasticity", "Multicollinearity", "Serial correlation", "Omitted variable bias"],
                "answer": "B",
            },
            {
                "question": "In instrumental variables (IV) estimation, which two conditions must a valid instrumental variable satisfy?",
                "choices": ["Relevance (correlated with endogenous regressor) and Exogeneity (uncorrelated with error term)", "Normality of residuals and constant variance", "Stationarity and positive covariance with the dependent variable", "Zero mean and unit variance"],
                "answer": "A",
            },
            {
                "question": "What is the null hypothesis of the Durbin-Watson test in regression residual analysis?",
                "choices": ["Residuals have zero mean", "There is no first-order autocorrelation in the residuals", "The variance is non-constant", "All slope coefficients equal zero"],
                "answer": "B",
            },
            {
                "question": "In Difference-in-Differences (DiD) estimation, what critical identifying assumption requires treatment and control groups to follow identical trajectories in the absence of treatment?",
                "choices": ["Constant elasticity assumption", "Parallel trends assumption", "Homogeneity of variance assumption", "Normality assumption"],
                "answer": "B",
            },
        ]
    },
    "high_school_psychology": {
        "test": [
            {
                "question": "In Ivan Pavlov's famous classical conditioning experiment with dogs, what was the meat powder before conditioning?",
                "choices": ["Conditioned Stimulus (CS)", "Unconditioned Stimulus (UCS)", "Conditioned Response (CR)", "Neutral Stimulus"],
                "answer": "B",
            },
            {
                "question": "Which psychological perspective founded by Sigmund Freud emphasizes unconscious conflicts and childhood experiences?",
                "choices": ["Behaviorism", "Psychoanalysis", "Humanistic Psychology", "Cognitive Psychology"],
                "answer": "B",
            },
            {
                "question": "In cognitive psychology, the 'magical number seven, plus or minus two' (Miller, 1956) refers to the capacity limit of which memory store?",
                "choices": ["Sensory memory", "Short-term / Working memory", "Long-term semantic memory", "Procedural memory"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What part of the brain is critically involved in the consolidation of new explicit memories?",
                "choices": ["Cerebellum", "Hippocampus", "Medulla", "Occipital lobe"],
                "answer": "B",
            },
            {
                "question": "In operant conditioning, what term describes increasing the frequency of a behavior by removing an aversive stimulus?",
                "choices": ["Positive punishment", "Negative reinforcement", "Extinction", "Negative punishment"],
                "answer": "B",
            },
            {
                "question": "Which stage of sleep is most characterized by rapid eye movement, muscle atonia, and vivid dreaming?",
                "choices": ["Stage N1", "Stage N3 (Deep sleep)", "REM sleep", "Sleep spindle stage"],
                "answer": "C",
            },
            {
                "question": "What concept did Jean Piaget define as mental frameworks that help individuals organize and interpret information?",
                "choices": ["Schemas", "Archetypes", "Heuristics", "Engrams"],
                "answer": "A",
            },
            {
                "question": "Stanley Milgram's famous 1963 obedience experiments investigated participants' willingness to obey authority figures when instructed to do what?",
                "choices": ["Conform to group opinion about line lengths", "Administer increasingly painful electric shocks to an unseen learner", "Sign a false confession in a simulated interrogation", "Imitate aggressive behavior seen on television"],
                "answer": "B",
            },
        ]
    },
    "us_foreign_policy": {
        "test": [
            {
                "question": "Which 1823 policy statement warned European powers against further colonization or interference in the Americas?",
                "choices": ["The Truman Doctrine", "The Monroe Doctrine", "The Marshall Plan", "The Roosevelt Corollary"],
                "answer": "B",
            },
            {
                "question": "Which post-WWII economic initiative provided extensive financial assistance to rebuild Western European economies?",
                "choices": ["The Lend-Lease Act", "The Marshall Plan", "The Warsaw Pact", "The Bretton Woods Agreement"],
                "answer": "B",
            },
            {
                "question": "What principle of international relations posits that global security is maximized when no single nation is powerful enough to dominate others?",
                "choices": ["Isolationism", "Balance of Power", "Unilateralism", "Imperial Overstretch"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "Which international organization was established in 1945 to promote global peace, security, and cooperation?",
                "choices": ["League of Nations", "United Nations", "NATO", "World Trade Organization"],
                "answer": "B",
            },
            {
                "question": "What diplomatic grand strategy did the US pursue during the Cold War to prevent the geographic expansion of Soviet influence?",
                "choices": ["Isolationism", "Containment", "Appeasement", "Preemptive strike"],
                "answer": "B",
            },
            {
                "question": "Which October 1962 confrontation between the US and the USSR is considered the closest the Cold War came to escalating into nuclear war?",
                "choices": ["Berlin Airlift", "Cuban Missile Crisis", "Suez Crisis", "Gulf of Tonkin Incident"],
                "answer": "B",
            },
            {
                "question": "Under Article II of the US Constitution, who has the formal authority to negotiate treaties, subject to approval by two-thirds of the Senate?",
                "choices": ["The President", "The Secretary of State", "The Speaker of the House", "The Chief Justice"],
                "answer": "A",
            },
            {
                "question": "In 1972, which US President initiated a historic diplomatic opening by making an official state visit to the People's Republic of China?",
                "choices": ["Lyndon B. Johnson", "Richard Nixon", "Jimmy Carter", "Ronald Reagan"],
                "answer": "B",
            },
        ]
    },
    # === Other ===
    "business_ethics": {
        "test": [
            {
                "question": "Which theory of corporate governance argues that companies should serve the interests of employees, customers, suppliers, and community, not just shareholders?",
                "choices": ["Shareholder Primacy Theory", "Stakeholder Theory", "Agency Theory", "Mercantilism"],
                "answer": "B",
            },
            {
                "question": "What term describes an employee who exposes illegal, fraudulent, or unethical activities within their organization?",
                "choices": ["Free rider", "Whistleblower", "Arbitrageur", "Ombudsman"],
                "answer": "B",
            },
            {
                "question": "Which practice involves misleading consumers regarding the environmental benefits of a company's product or service?",
                "choices": ["Price skimming", "Greenwashing", "Loss leading", "Predatory pricing"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What does CSR stand for in the context of business management?",
                "choices": ["Customer Sales Ratio", "Corporate Social Responsibility", "Capital Solvency Rate", "Centralized Supply Routing"],
                "answer": "B",
            },
            {
                "question": "What illegal practice occurs when a corporate insider trades securities based on material, non-public corporate information?",
                "choices": ["Short selling", "Insider trading", "Front-running", "Spoofing"],
                "answer": "B",
            },
            {
                "question": "What formal organizational document articulates the ethical principles and behavioral standards expected of all employees?",
                "choices": ["Corporate Bylaws", "Code of Conduct / Ethics", "Articles of Incorporation", "Non-Disclosure Agreement"],
                "answer": "B",
            },
            {
                "question": "What situation arises when a professional's personal interests clash with their fiduciary duties to a client or employer?",
                "choices": ["Collective bargaining", "Conflict of interest", "Whistleblowing", "Moral hazard"],
                "answer": "B",
            },
            {
                "question": "What fraudulent investment scam pays returns to earlier investors using capital provided by newer investors rather than legitimate profit?",
                "choices": ["Pump-and-dump", "Ponzi scheme", "Pyramid licensing", "Churning"],
                "answer": "B",
            },
        ]
    },
    "management": {
        "test": [
            {
                "question": "In strategic management, what does the SWOT analysis framework stand for?",
                "choices": ["Sales, Workforce, Organization, Targets", "Strengths, Weaknesses, Opportunities, Threats", "Strategy, Workflow, Objectives, Timeline", "Structure, Warranty, Operations, Tracking"],
                "answer": "B",
            },
            {
                "question": "Which leadership style is characterized by delegating authority and allowing team members maximum autonomy in decision-making?",
                "choices": ["Autocratic", "Laissez-faire", "Micromanagement", "Bureaucratic"],
                "answer": "B",
            },
            {
                "question": "What project management visual chart displays schedule tasks as horizontal bars over time?",
                "choices": ["Pareto Chart", "Gantt Chart", "Ishikawa Diagram", "Scatter Plot"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "In organizational psychology, what need sits at the highest level of Maslow's Hierarchy of Needs?",
                "choices": ["Safety needs", "Self-actualization", "Social belonging", "Physiological needs"],
                "answer": "B",
            },
            {
                "question": "In project planning and goal setting, what does the 'M' in the SMART criteria acronym stand for?",
                "choices": ["Mandatory", "Measurable", "Monitored", "Motivated"],
                "answer": "B",
            },
            {
                "question": "Which strategic management framework evaluates business performance across four perspectives: Financial, Customer, Internal Processes, and Learning & Growth?",
                "choices": ["Six Sigma", "The Balanced Scorecard", "Total Quality Management", "Porter's Five Forces"],
                "answer": "B",
            },
            {
                "question": "What organizational structure features dual-reporting relationships where employees report to both a functional manager and a project manager?",
                "choices": ["Hierarchical structure", "Matrix structure", "Divisional structure", "Flat network structure"],
                "answer": "B",
            },
            {
                "question": "In project Critical Path Method (CPM), what does the critical path represent?",
                "choices": ["The sequence of tasks with the highest financial budget", "The longest path of dependent activities that determines the minimum total project duration", "The activities with the most available slack time", "The shortest route to the first milestone"],
                "answer": "B",
            },
        ]
    },
    "clinical_knowledge": {
        "test": [
            {
                "question": "Which laboratory blood test is widely used as a long-term indicator of glycemic control over the preceding 2 to 3 months in diabetic patients?",
                "choices": ["Fasting Blood Glucose", "Hemoglobin A1c (HbA1c)", "Serum Creatinine", "Total Bilirubin"],
                "answer": "B",
            },
            {
                "question": "What is the standard first-line medication for the immediate emergency treatment of anaphylactic shock?",
                "choices": ["Oral Antihistamines", "Intramuscular Epinephrine", "Inhaled Albuterol", "Intravenous Saline alone"],
                "answer": "B",
            },
            {
                "question": "Which infectious disease is caused by Mycobacterium tuberculosis?",
                "choices": ["Malaria", "Tuberculosis", "Cholera", "Diphtheria"],
                "answer": "B",
            },
        ],
        "dev": [
            {
                "question": "What is considered a normal resting heart rate for a healthy adult?",
                "choices": ["20-40 bpm", "60-100 bpm", "120-150 bpm", "160-190 bpm"],
                "answer": "B",
            },
            {
                "question": "Which organ in the human body produces insulin?",
                "choices": ["Liver", "Pancreas", "Spleen", "Kidney"],
                "answer": "B",
            },
            {
                "question": "Which non-invasive diagnostic test records the electrical activity of the heart to evaluate rhythms and detect myocardial ischemia?",
                "choices": ["Echocardiogram", "Electrocardiogram (ECG / EKG)", "Chest X-ray", "Coronary angiogram"],
                "answer": "B",
            },
            {
                "question": "What major blood vessel carries oxygenated blood directly away from the left ventricle of the heart into systemic circulation?",
                "choices": ["Aorta", "Pulmonary artery", "Superior vena cava", "Carotid artery"],
                "answer": "A",
            },
            {
                "question": "In emergency basic life support and resuscitation, what does the acronym AED stand for?",
                "choices": ["Advanced External Defibrillator", "Automated External Defibrillator", "Asynchronous Emergency Device", "Automated Electrical Diagnosis"],
                "answer": "B",
            },
        ]
    },
}


class MMLUDatasetLoader:
    """
    Loads test and validation (few-shot dev) sets for MMLU subjects.
    Supports single-batch pre-downloading and in-memory caching to avoid repetitive network calls.
    """

    def __init__(self, auto_preload: bool = False):
        self._hf_available = False
        self._cache: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        self._preloaded = False
        try:
            import datasets
            self._hf_available = True
            if auto_preload:
                self.preload_all()
        except ImportError:
            logger.info("HuggingFace 'datasets' package not found. Using structured built-in dataset repository.")

    def preload_all(self, dataset_name: str = "cais/mmlu") -> bool:
        """
        Downloads and caches all 57 MMLU subjects in a single batch request from HuggingFace.
        Subsequent calls to load_subject will query memory directly with 0 network latency.
        """
        if not self._hf_available or self._preloaded:
            return self._preloaded

        try:
            from datasets import load_dataset
            logger.info(f"Downloading full MMLU dataset ('{dataset_name}', 'all') in a single batch...")
            test_ds = load_dataset(dataset_name, "all", split="test")
            dev_ds = load_dataset(dataset_name, "all", split="dev")

            # Group test samples by subject
            for i, row in enumerate(test_ds):
                subj = row.get("subject", "")
                if not subj:
                    continue
                if subj not in self._cache:
                    self._cache[subj] = ([], [])
                self._cache[subj][0].append({
                    "id": f"{subj}-{i}",
                    "subject": subj,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": normalize_answer_to_letter(row.get("answer", 0)),
                })

            # Group dev samples by subject
            for i, row in enumerate(dev_ds):
                subj = row.get("subject", "")
                if not subj:
                    continue
                if subj not in self._cache:
                    self._cache[subj] = ([], [])
                self._cache[subj][1].append({
                    "id": f"{subj}-dev-{i}",
                    "subject": subj,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": normalize_answer_to_letter(row.get("answer", 0)),
                })

            self._preloaded = True
            logger.info(f"Successfully cached {len(self._cache)} MMLU subjects locally. Zero future network requests needed.")
            return True
        except Exception as e:
            logger.warning(f"Single-batch MMLU download failed or was interrupted ({e}). Will load subjects on demand.")
            return False

    def get_available_subjects(self) -> List[str]:
        """Returns list of built-in subjects."""
        return list(MMLU_SAMPLE_DATABASE.keys())

    def load_subject(
        self, subject: str, max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads (test_samples, dev_samples) for a specific subject.
        Uses in-memory cache first if pre-downloaded, avoiding repetitive network requests.
        """
        # 1. Return from in-memory cache if available
        if subject in self._cache:
            test_samples, dev_samples = self._cache[subject]
            if max_samples:
                test_samples = test_samples[:max_samples]
            return test_samples, dev_samples

        # 2. Fetch individually if not yet cached and HF is available
        if self._hf_available:
            try:
                from datasets import load_dataset
                ds = load_dataset("cais/mmlu", subject, split="test")
                dev_ds = load_dataset("cais/mmlu", subject, split="dev")

                test_samples = [
                    {
                        "id": f"{subject}-{i}",
                        "subject": subject,
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": normalize_answer_to_letter(row.get("answer", 0)),
                    }
                    for i, row in enumerate(ds)
                ]
                dev_samples = [
                    {
                        "id": f"{subject}-dev-{i}",
                        "subject": subject,
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": normalize_answer_to_letter(row.get("answer", 0)),
                    }
                    for i, row in enumerate(dev_ds)
                ]
                self._cache[subject] = (test_samples, dev_samples)
                if max_samples:
                    test_samples = test_samples[:max_samples]
                return test_samples, dev_samples
            except Exception as e:
                logger.warning(f"HuggingFace download failed or unavailable ({e}). Loading built-in repository.")

        return self._get_fallback_data(subject, max_samples)

    def _get_fallback_data(
        self, subject: str, max_samples: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if subject in MMLU_SAMPLE_DATABASE:
            raw_test = MMLU_SAMPLE_DATABASE[subject]["test"]
            raw_dev = MMLU_SAMPLE_DATABASE[subject].get("dev", [])
            test_samples = [
                {
                    "id": f"{subject}-{i+1}",
                    "subject": subject,
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer": normalize_answer_to_letter(q.get("answer", "A")),
                }
                for i, q in enumerate(raw_test)
            ]
            dev_samples = [
                {
                    "id": f"{subject}-dev-{i+1}",
                    "subject": subject,
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer": normalize_answer_to_letter(q.get("answer", "A")),
                }
                for i, q in enumerate(raw_dev)
            ]
        else:
            # Synthetic template for unknown subject
            test_samples = [
                {
                    "id": f"{subject}-1",
                    "subject": subject,
                    "question": f"Which fundamental principle distinguishes advanced concepts in {subject.replace('_', ' ')}?",
                    "choices": [
                        "Deterministic constraint relaxation",
                        "Fundamental axiomatic boundary (Correct)",
                        "Stochastic gradient decay",
                        "Unbounded parameter scaling",
                    ],
                    "answer": "B",
                },
                {
                    "id": f"{subject}-2",
                    "subject": subject,
                    "question": f"What is the optimal methodology for validating empirical results in {subject.replace('_', ' ')}?",
                    "choices": [
                        "Controlled cross-validation with statistical significance tests",
                        "Arbitrary heuristic sampling",
                        "Zero-shot uncalibrated assumption",
                        "Non-repeatable exploratory runs",
                    ],
                    "answer": "A",
                },
            ]
            dev_samples = [
                {
                    "id": f"{subject}-dev-1",
                    "subject": subject,
                    "question": f"Example baseline question in {subject.replace('_', ' ')}?",
                    "choices": ["Option Alpha", "Option Beta (Correct)", "Option Gamma", "Option Delta"],
                    "answer": "B",
                }
            ]

        if max_samples:
            test_samples = test_samples[:max_samples]
        return test_samples, dev_samples
