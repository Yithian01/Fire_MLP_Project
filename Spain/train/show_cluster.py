import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from colorcet import fire  # 예쁜 색상

df = pd.read_csv('./result/example_clustered.csv', encoding='utf-8', engine='python')

canvas = ds.Canvas(plot_width=1000, plot_height=800)
agg = canvas.points(df, 'x', 'y', agg=ds.count_cat('cluster'))
img = tf.shade(agg, color_key=None, how='eq_hist')
export_image(img, filename='clustering_result_ds', background="white")
