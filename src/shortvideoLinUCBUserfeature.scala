import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.clustering.{KMeans,KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler

object shortvideoLinUCBUserfeature {
  val spark = SparkSession.builder.appName("shortvideoLinUCBUserfeature").enableHiveSupport().getOrCreate()
  import spark.implicits._
  import spark.sql

  def loaddata(dt:String)={
    val load_sql = s"select * from temp.shortvideo_user_matrix where dt='$dt'"
    val data = sql(load_sql)
    data
  }

  def feature(data:DataFrame)={
    val cols = data.drop("kgid","dt").columns

    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val assembled_df = assembler.transform(data).select("kgid","features")
    assembled_df
  }

  def train(assembled_df:DataFrame): Unit ={
    val Array(train,test)=assembled_df.randomSplit(Array(0.1,0.9))

    val k = 10
    val MaxIter = 20
    val kmeans = new KMeans().setK(k).setMaxIter(MaxIter).setFeaturesCol("features").setPredictionCol("label")
    val clusters = kmeans.fit(train)

    val model_path = "/user/LinUCB/model"
    clusters.write.overwrite().save(model_path + "/LinUCBclusters.model")
    println("模型保存完成")
  }

  def cluster(assembled_df:DataFrame,dt:String): Unit ={
    val model_path = "/user/LinUCB/model"
    val clusters = KMeansModel.load(model_path + "/LinUCBclusters.model")

    val results = clusters.transform(assembled_df)

    results.createOrReplaceTempView("results")

    // 分词结果插入hive表中
    val insert_sql=s"insert overwrite table temp.shortvideo_user_matrix_aftercluster partition(dt='$dt') select kgid,features,label from results"
    spark.sql(insert_sql)
  }

  def main(): Unit ={
    val dt="2019-01-24"
    val data = loaddata(dt)
    val assembled_df = feature(data)
    //train(assembled_df)
    cluster(assembled_df,dt)
  }
}
