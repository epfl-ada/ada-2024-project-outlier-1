import numpy as np

import plotly.graph_objects as go

def generate_performance_scatter(avglen,models=['llama3_2007','mistral_2007']):
    """
    Generate a scatter plot of the performance of the models

    Args:
        avglen (pandas dataframe): dataframe containing the average length of the path for each model
        models (list): list of models to compare (max 2)
    """
    fig = go.Figure()

    colors = ['red', 'green', 'blue', 'orange']
    # Add model performance
    for i,model in enumerate(models):
        fig.add_trace(go.Scatter(
            x=avglen[model],
            y=np.arange(len(avglen[model])),
            mode='markers',
            name=model + ' performance',
            marker=dict(color=colors[i], symbol='x')
        ))
    
    # Add Player mean with error bars
    fig.add_trace(go.Scatter(
    x=avglen['mean'],
    y=np.arange(len(avglen['mean'])),
    mode='markers',
    name='Player mean',
    opacity=0.8,
    marker=dict(color='LightSkyBlue', symbol='circle'),
    error_x=dict(
        type='data',
        array=2*avglen['std'],
        visible=True
    )
))

# Customize the layout
    fig.update_layout(
    title='Average path length',
    title_x=0.5,
    xaxis_title='Length',
    yaxis_title='Source -> Target',
    template='plotly_white',
    yaxis=dict(
        tickmode='array',
        tickvals=np.arange(len(avglen.index)),
        ticktext=avglen[['start', 'end']].apply(lambda x: x[0] + ' -> ' + x[1], axis=1)
    ),
    height=2000,
    width=1000
)

    fig.show()