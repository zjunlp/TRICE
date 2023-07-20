math_sys_prompt = '''I will provide you with a math Question and a Golden answer. I need you to write "calculator(formula)" to invoke the API for assistance in solving the question, where "formula" is the formula to reach the Golden answer. Here are some examples:'''

math_examples = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "golden_answer": ["72"],
        "output": "calculator(48+48/2)"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "golden_answer": ["10"],
        "output": "calculator((12/60)*50)"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "golden_answer": ["5"],
        "output": "calculator(100-100/2-15-15*2)"
    }
]


lama_sys_prompt = '''I will provide you with a Question, Golden answers. I need you to write "QA(question)" to invoke the API for assistance in answering the Question, where "question" is the question you want to ask to obtain the Golden answers. Here are some examples:'''


trex_examples = [
    {
        "question": "The army held Rome for a brief time, but was then forced to retreat to the city of Perusia (modern Perugia, ?",
        "golden_answer": ["Italy"],
        "output": "QA(Which country is Perusia, or modern Perugia, located in?)",
    },
    {
        "question": "Winners of the festivals «Chervona Ruta» (Ukraine), «Pearls of the Season» (Ukraine), «Boards» (Moscow), «Woodstock» (?",
        "golden_answer": ["Poland"],
        "output": "QA(Where is the Woodstock festival held?)",
    },
    {
        "question": "It is native to the Alps and the Pyrenees Mountains of Europe (Spain, France, Italy, Switzerland, Austria and ?",
        "golden_answer": ["Germany"],
        "output": "QA(Which country is mentioned as being native to the Alps and the Pyrenees Mountains alongside Spain, France, Italy, Switzerland, and Austria?)",
    },
    {
        "question": "Heorhiy Kyrylovych Tkachenko (May 5, 1898 in Hlushkovo, Kursk region of the Russian Empire – 1993 in Kiev, ?",
        "golden_answer": ["Ukraine"],
        "output": "QA(Where did Heorhiy Kyrylovych Tkachenko die?)",
    },
]

qa_sys_prompt = '''I will provide you with a Question, Golden answers. I need you to write "WikiSearch(term)" to invoke the API for assistance in answering the Question, where "term" is the search term you want to look up to obtain the Golden answers. Here are some examples:'''

webq_examples = [

    {
        "question": "what is nina dobrev nationality?",
        "golden_answer": ["Bulgaria"],
        "output": "WikiSearch(Nina Dobrev nationality)",
    },
    {
        "question": "what type of car does michael weston drive?",
        "golden_answer": ["Wishcraft"],
        "output": "WikiSearch(Michael Weston car)",
    },
    {
        "question": "where are sunbeam microwaves made?",
        "golden_answer": ["Florida"],
        "output": "WikiSearch(Sunbeam microwaves manufacturing location)",
    },
    {
        "question": "what religion are people in russia?",
        "golden_answer": ["Islam", "Russian Orthodox Church"],
        "output": "WikiSearch(Religion in Russia)",
    },
]

nq_examples = [
    {
        "question": "who plays penny pinchelow in dumb and dumber to",
        "golden_answer": ["Rachel Melvin"],
        "output": "WikiSearch(Penny Pinchelow Dumb and Dumber To)",
    },
    {
        "question": "when is the next episode of the flash airing",
        "golden_answer": ["October 9 , 2018"],
        "output": "WikiSearch(The Flash next episode air date)",
    },
    {
        "question": "who has the most wins in an mlb season",
        "golden_answer": ["Seattle Mariners",  "Chicago Cubs"],
        "output": "WikiSearch(MLB team with most wins in a season)",
    },
    {
        "question": "who made the one ring to rule them all",
        "golden_answer": ["Sauron the Dark Lord"],
        "output": "WikiSearch(creator of the one ring in Lord of the Rings)",
    },
]

tqa_examples = [
    {
        "question": "What are oysters wrapped in bacon called?",
        "golden_answer": ["Angels on horseback", "Angels on Horseback"],
        "output": "WikiSearch(oysters wrapped in bacon name)",
    },
    {
        "question": "Who won the F.I.M. World Championship 500cc motocross title in 1996?",
        "golden_answer": ["Shayne King"],
        "output": "WikiSearch(F.I.M. World Championship 500cc motocross 1996 winner)",
    },
    {
        "question": "How many kings of France were named Louis?",
        "golden_answer": ["eighteen", "18"],
        "output": "WikiSearch(Kings of France named Louis)",
    },
    {
        "question": "How did Jock die in Dallas?",
        "golden_answer": ["Helicopter accident"],
        "output": "WikiSearch(Jock Ewing death)",
    },
]


examples_dict = {
    "ASDiv": math_examples,
    "SVAMP": math_examples,
    "GSM8K": math_examples,
    "T-REx": trex_examples,
    "WebQ": webq_examples,
    "NaturalQ": nq_examples,
    "TriviaQA": tqa_examples,
}

sys_dict = {
    "ASDiv": math_sys_prompt,
    "SVAMP": math_sys_prompt,
    "GSM8K": math_sys_prompt,
    "T-REx": lama_sys_prompt,
    "WebQ": qa_sys_prompt,
    "NaturalQ": qa_sys_prompt,
    "TriviaQA": qa_sys_prompt,
}
