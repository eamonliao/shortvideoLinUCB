from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

conf = (SparkConf().setAppName("goodsinger_model_fit"))
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def load_data():
    df = spark.sql("select * from fx_dal.shortvideo_user_behave_stat_wide_30d where dt='2019-01-20'")
    df = df.toPandas()
    return df

# 训练数据特征归一化
def scale_data(X_data):
    limit_log_cols=["starlevel","richlevel","live_count","fans_cusum","fans_count","consume_user_count"
        ,"get_bean","real_sing_count","song_num","sing_score_mean","sing_score_median","sing_like_count"
        ,"user_mv_count","self_mv_count","user_mv_num","mv_valid_play_count","mv_like_count"
        ,"song_order_count","song_order_user_num","song_order_bean","chat_num","word_num1","word_num2","word_num3"
        ,"word_num4","word_num5","word_num6","word_num7","word_num8","word_num9","word_num10","song_key_fans_num"
        ,"sing_follow_count","sing_gift_user_num","sing_gift_count","sing_gift_coin","sing_gift_bean","sing_listen_num"
        ,"sing_out_count","sing_out_rate","is_yueqi","is_dj","masterpk_num","masterwin_num","masterwin_rate"
        ,"competitorpk_num","competitorwin_num","competitorwin_rate","nofans_enternum","nofans_outernum"
        ,"nofans_outrate","nofans_sing_gift_user_num","nofans_sing_gift_count","nofans_sing_gift_coin"
        ,"nofans_sing_gift_bean","user_mv_count_nofans","user_mv_num_nofans","consume_user_count_nofans"
        ,"get_bean_nofans","song_order_user_num_nofans","song_order_count_nofans","song_order_bean_nofans"
        ,"chat_num_nofans","word_num1_nofans","word_num2_nofans","word_num3_nofans","word_num4_nofans"
        ,"word_num5_nofans","word_num6_nofans","word_num7_nofans","word_num8_nofans","word_num9_nofans"
        ,"word_num10_nofans","live_duration","like_num_per_song","mv_count_per_fan"
        ,"song_order_per_fan","song_order_sing_rate","key_fans_rate","new_fans_rate","song_order_bean_rate"
        ,"song_order_bean_rate_nofans","word1_rate","word2_rate","word3_rate","word4_rate","word5_rate"
        ,"word10_rate","word1_rate_nofans","word2_rate_nofans"
        ,"word4_rate_nofans","word5_rate_nofans"
        ,"word10_rate_nofans"]
    X_data=X_data.copy()
    for col in limit_log_cols:
        if col not in X_data.columns:
            continue
        limit_value=X_data[col].quantile(0.99)
        X_data[col]=X_data[col].apply(lambda x:np.log(min(x,limit_value)+1))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scale_data=scaler.fit_transform(X_data)
    return scale_data

