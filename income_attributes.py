ATTR_TYPES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country'
]

CONTINUOUS_ATTRS = {
    'age': 'Age',
    'fnlwgt': 'Final-Weight',
    'education-num': 'Education-Num',
    'capital-gain': 'Capital-Gain',
    'capital-loss': 'Capital-Loss',
    'hours-per-week': 'Hours-Per-Week',
}

ATTRS = [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked',
    'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool',
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse',
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces',
    'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried',
    'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black',
    'Female', 'Male',
    'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China','Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands',
    'Age','Final-Weight','Education-Num','Capital-Gain','Capital-Loss','Hours-Per-Week'
]

ATTR_TO_IDX = {k:v for v,k in enumerate(ATTRS)}

LABELS = {
    '>50K': 1.0,
    '<=50K': 0.0
}

