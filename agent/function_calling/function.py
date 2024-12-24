import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import io
from PIL import Image


def barchart( **kwargs):
    return sns.barplot(**kwargs)


def linechart(**kwargs):
    return sns.lineplot(**kwargs)


def scatterplot(**kwargs):
    return sns.scatterplot(**kwargs)

def get_single_chart(type_, params, df, ax = None):
    
    if "bar" in type_:
        return barchart(ax=ax, data=df, width=0.5, **params)
    
    elif "line" in type_:
        return linechart(ax=ax, data=df, linewidth = 2.5, **params)
    
    elif "scatter" in type_:
        return scatterplot(ax=ax, data=df, **params)



def create_chart(params):
    
    df = pd.DataFrame(params['data'])
    
    key_ = list(params['chart'].keys())
    num_chart = len(key_)
            
    print(df)
    
    
    if "title" in params.keys():
        plt.title(params["title"])
    
    if num_chart == 1:
        
        key_1 = key_[0]
            
        get_single_chart(key_, params['chart'][key_1], df=df)
                
    elif num_chart == 2:
        

        plt.figure(figsize=(8, 6))
        key_1 = key_[0]
        key_2 = key_[1]
        
        df[params['chart'][key_1]['x']] = df[params['chart'][key_1]['x']].astype(str)
        
        ax = get_single_chart(key_1, params['chart'][key_1], df=df.copy())
        ax2 = ax.twinx()
        get_single_chart(key_2, params['chart'][key_2], df=df.copy(), ax=ax2)
        
        # Set the legend for the first and second chart. Legend color for title of each chart
        
        color1 = params['chart'][key_1].get('color', 'skyblue')
        color2 = params['chart'][key_2].get('color', 'orange')
        
        label1 = params['chart'][key_1].get('y', key_1)
        label2 = params['chart'][key_2].get('y', key_2)
        
        legend_elements = [
            mpatches.Patch(color= color1, label=label1),
            mpatches.Patch(color=color2, label=label2)
        ]

        plt.legend(handles=legend_elements)  
    
    else:
        raise ValueError("Only 1-2 charts are supported")
        
    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, format='png')
    plt.close()
    
    image_buffer.seek(0)
    
    return image_buffer
    
    
    
    
    
    
if __name__ == "__main__":
    import pandas as pd
    data = {
        "data": {
                "year": [2020, 2021, 2022, 2023],
                "ROA": [0.5, 0.3, 0.7, 0.2],
                "ROE": [0.8, 0.7, 1, 0.5]
        },
        "chart": {
            "barchart": {
                "x": "year",
                "y": "ROA",
                "color": "skyblue",
            },
            "linechart": {
                "x": "year",
                "y": "ROE",
                "color": "red"
            }
        },
        "title": "ROA and ROE of NEU"
    }
    
    chart_image = create_chart(data)
    
    pil_image = Image.open(chart_image)
    
    pil_image.show()

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import numpy as np
    # import pandas as pd

    # # Create example data
    # data1 = pd.DataFrame({
    #     "x": np.linspace(0, 10, 100),
    #     "y": np.sin(np.linspace(0, 10, 100))
    # })

    # data2 = pd.DataFrame({
    #     "x": np.linspace(0, 10, 100),
    #     "y": np.cos(np.linspace(0, 10, 100))
    # })

    # # Create the plot
    # plt.figure(figsize=(8, 6))
    # ax = sns.lineplot(data=data1, x="x", y="y", label="sin(x)")
    # sns.lineplot(data=data2, x="x", y="y", label="cos(x)", ax=ax)

    # # Customize the plot
    # ax.set_title("Overlayed Line Plots")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.legend()

    # plt.show()