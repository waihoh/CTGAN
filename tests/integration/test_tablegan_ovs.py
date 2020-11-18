import matplotlib.pyplot as plt
import seaborn as sns
from ctgan.tablegan import TableganSynthesizer
from ctgan.transformer import DataTransformer
import pandas as pd
import numpy as np
from pathlib import Path

# Load cleaned 2016-2018 OVS datasets
OVS1618 = pd.read_csv('C:/Users/stazt/Documents/nBox/Project Ultron/Tianming/Dataset/OVS1618v4_cleaned.csv')
# List of all categorical variables
with open('C:/Users/stazt/Documents/nBox/Project Ultron/Tianming/Dataset/categorical_columns.txt', 'r') as f:
    cat_cols = f.read().split('\n')[:-1]
    f.close()

# List of 22 expenditure variables
cont_cols = ['Expenditure on Accommodation (c4.tot.new)',
             'Expenditure on Hotels (c4d_1.r + c4d_2.r + c4d_3.r)',
             'Expenditure on Service Apartment (c4d_6.r)',
             'Expenditure on Other Accommodations (c4d_4.r + c4d_5.r + c4d_7.r + c4d_8.r)',
             'Expenditure on F&B (c6.tot.new)',
             'Expenditure on Hawker Centre, Food Court or Coffee Shop (c6c_1.r)',
             'Expenditure on Casual Dining (c6c_2.r)',
             'Expenditure on Fine-dining, Celebrity Chef Restaurants (c6c_3.r)',
             'Expenditure on Transport (c7.tot.new)',
             'Expenditure on Sightseeing & Entertainment (c10.tot.new)',
             'Expenditure on Sightseeing (c10c_1.r)',
             'Expenditure on Attractions (c10c_2.r)',
             'Expenditure on Entertainment or Nightspots (c10c_3.r)',
             'Expenditure on Business (c11.tot.new)',
             'Expenditure on Education (c12c_1.r)',
             'Expenditure on Healthcare (c12c_2.r + c12c_3.r + c12c_4.r + c12c_5.r + c12c_6.r)',
             'Expenditure on Shopping (t7.m.any)',
             'Expenditure on Healthcare & Wellness Products (t7.m.well)',
             'Expenditure on Confectionery & Food Items (t7.m.food)',
             'Expenditure on Fashion, Jewellery & Watches (t7.m.fash + t7.m.jew + t7.m.wat)',
             'Expenditure on Other Shopping Items (t7.m.gift + t7.m.ctec + t7.m.anti + t7.m.oth)',
             'Package Expenditure Per Person (c1b.r)']

# Reduce OVS dataset to the 22 continuous benchmark columns
ovs = OVS1618[cont_cols]


# Track rows for each continuous column that contains NA values; 1 if NA, 0 is non-NA
def na_indicator(df, list_cols):
    '''
    Inserts a binary categorical column to indicate if a continuous variable contains NA values.

    Args:
        df (Dataframe): dataframe to perform function on
        list_cols (list): list of columns

    Returns:
        output (dictionary): dictionary of (keys) columns and (values) list of its rows with NA values
    '''
    output = {}
    for col in list_cols:
        na_cols = []
        for i in range(len(df)):
            if not df[col][i] > 0:
                na_cols.append(i)
        output[col] = na_cols
    return output


ovs_na = na_indicator(ovs, cont_cols)

# Replace all NA values by -999
ovs2 = ovs.fillna(-999)


def plot_hist(ovs, ovs_vgm, ovs_norm, cont_cols):
    Path('C:/Users/stazt/Documents/nBox/Project Ultron/Tianming/').mkdir(parents=True, exist_ok=True)  # save plots
    for col in cont_cols:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))

        fig.suptitle(col)

        ax[0, 0].hist(ovs[col])
        ax[0, 0].set_title('Original')

        sns.kdeplot(ovs[col], shade=True, ax=ax[1, 0], legend=False)
        ax[1, 0].set_title('Original')

        ax[0, 1].hist(ovs_vgm[col])
        ax[0, 1].set_title('VGM Transformation')

        sns.kdeplot(ovs_vgm[col], shade=True, ax=ax[1, 1], legend=False)
        ax[1, 1].set_title('VGM Transformation')

        ax[0, 2].hist(ovs_norm[col])
        ax[0, 2].set_title('Min-Max Normalisation')

        sns.kdeplot(ovs_norm[col], shade=True, ax=ax[1, 2], legend=False)
        ax[1, 2].set_title('Min-Max Normalisation')

        plt.show()
        fig.savefig(f'C:/Users/stazt/Documents/nBox/Project Ultron/Tianming/{col}.png')

###VGM transformer
tablegan = TableganSynthesizer()
###min-max transformer
print('Training tablegan is starting')
tablegan.fit(ovs2, discrete_columns=tuple(), epochs=1, model_summary=True,trans="Min-Max")
print('Training tablegan is completed')
samples_minmax = tablegan.sample(ovs2.shape[0])
## use VGM transformation
#tablegan = TableganSynthesizer()
print('Training tablegan is starting')
tablegan.fit(ovs2, discrete_columns=tuple(), epochs=1, model_summary=True,trans="VGM")
samples_vgm = tablegan.sample(ovs2.shape[0])
print('Training tablegan is completed')

# Comparison of distributions: OVS datasets, VGM transformation, and min-max normalisation
plot_hist(ovs, samples_vgm, samples_minmax, cont_cols)

##compare mean
compare_mean = np.vstack((ovs2.mean(),samples_vgm.mean(),samples_minmax.mean()))
