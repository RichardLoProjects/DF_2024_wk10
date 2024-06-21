print('Importing libraries...')
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest, expon
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")




print('Initialising script...')
'''Key variables'''
card_columns = [
    'name'
    , 'real_name'
    , 'faction_name'
    , 'type_name'
    , 'health'
    , 'sanity'
    , 'position'
    , 'traits'
    , 'skill_willpower'
    , 'skill_intellect'
    , 'skill_combat'
    , 'skill_agility'
    , 'flavor'
]
my_local_cards = dict()
df_cards = pd.DataFrame(columns=card_columns)
row_count = 325




'''Functions'''
def get_card_obj(card_id):
    '''Use API to return a single card object to be cached locally.'''
    try:
        url = f'https://arkhamdb.com/api/public/card/0{card_id}.json'
        response = requests.get(url)
        card = response.json()
        return card
    except:
        return 'Error'


def get_card_cols(card_id):
    '''Places a card as a row into the dataframe.'''
    try:
        _temp_list = card_columns
        return tuple([my_local_cards[card_id].get(_name, np.NaN) for _name in _temp_list])
    except:
        return tuple(['Error'] + [np.NaN for _ in range(len(_temp_list[1:]))])


def is_this_exp_dist(data):
    '''Tests if data is exponentially distributed.'''
    # Estimate parameters and perform test
    lambda_est = 1 / np.mean(data)
    cdf = lambda x: expon.cdf(x, scale=1/lambda_est)
    ks_statistic, p_value = kstest(data, cdf)
    # Display the results
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")
    if p_value > 0.05:
        print("Fail to reject the null hypothesis: Data follows the exponential distribution.")
        return True
    else:
        print("Reject the null hypothesis: Data does not follow the exponential distribution.")
        return False



    
'''Create Dataframe'''
# Cache data locally
print('Hitting the API (please be patient)...')
_logging_progress = 0
_logged_prog = set()
for card_idx in range(2001, 2001+row_count):
    my_local_cards[card_idx] = get_card_obj(card_idx)
    _logging_progress += 100
    _logged_percent = (_logging_progress/row_count)//1
    if _logged_percent%5 == 0 and _logged_percent not in _logged_prog:
        print(f'{_logged_percent}%')
        _logged_prog.add(_logged_percent)

# Insert data into dataframe
print('Making csv...')
for row_num in range(row_count):
    df_cards.loc[row_num] = get_card_cols(2001+row_num)
df_cards = df_cards.set_index('position')
df_cards.to_csv('arkham_card.csv')




'''Visualisation'''
print('Plotting graphs...')
# Count of factions
sns.countplot(x='faction_name', data=df_cards)
plt.title('Faction Populations')
plt.savefig('faction_count.png', dpi = 300, bbox_inches = 'tight')

# Count of types
sns.countplot(x='type_name', data=df_cards.sort_values('type_name', ascending=True))
plt.title('Popluation of Types (alphabetically)')
plt.xticks(
    rotation=45,
    rotation_mode = 'anchor',
    horizontalalignment = 'right',
    fontweight = 'light',
    fontsize = 'large'
)
plt.savefig('type_count.png', dpi = 300, bbox_inches = 'tight')

# Count of traits with a count of more than 5
_temp_data = df_cards[df_cards['traits'].isin(df_cards['traits'].value_counts()[df_cards['traits'].value_counts() > 5].index)]
sns.countplot(x='traits', data=_temp_data, order=_temp_data['traits'].value_counts(ascending=False).index)
plt.title('Popluation of Traits')
plt.xticks(
    rotation=45,
    rotation_mode = 'anchor',
    horizontalalignment = 'right',
    fontweight = 'light',
    fontsize = 'large'
)
plt.savefig('trait_count.png', dpi = 300, bbox_inches = 'tight')

# Health distribution
sns.displot(
    df_cards['health'].dropna(inplace=False),
    height = 8,
    #bins=50,
    kde=True,
    color='blue'
)
plt.title('Healthy Histogram')
plt.savefig('hist_hp.png', dpi = 300, bbox_inches = 'tight')

# Sanity distribution
sns.displot(
    df_cards['sanity'].dropna(inplace=False),
    height = 8,
    #bins=50,
    kde=True,
    color='blue'
)
plt.title('Sanity Histogram')
plt.savefig('hist_sanity.png', dpi = 300, bbox_inches = 'tight')

# Statistical testing
print('\n' + 'Statistical tests...')
is_this_exp_dist(df_cards['health'].dropna(inplace=False))
is_this_exp_dist(df_cards['sanity'].dropna(inplace=False))
print(' ')

# Pairplots
sns.pairplot(
    df_cards[['health', 'sanity', 'skill_willpower', 'skill_intellect', 'skill_combat', 'skill_agility']].dropna(inplace=False),
    height = 6
)
plt.savefig('pairplot.png', dpi = 300, bbox_inches = 'tight')

# Heatmap of correlation matrix
card_corr = df_cards[['health', 'sanity', 'skill_willpower', 'skill_intellect', 'skill_combat', 'skill_agility']].dropna(inplace=False).corr()
plt.figure(figsize = (15,15))
sns.heatmap(
    card_corr,
    annot = True,
    vmin=-1,
    vmax=1,
    cmap=sns.color_palette("YlOrBr_r", as_cmap=True)
)
plt.savefig('heatmap.png', dpi = 300, bbox_inches = 'tight')

print('Task complete!\n\n')
