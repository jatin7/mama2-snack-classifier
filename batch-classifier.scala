// this file is meant to be modified and loaded at the spark-shell using
// :load file-name

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel

val data = sc.textFile("/user/<example>/cereal.dat")
val trainingData = data.map { line =>
  val parts = line.split('\t')
  LabeledPoint(parts(5).toDouble, Vectors.dense(parts.slice(0, 5).map(_.toDouble)))
}
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int](0 -> 2, 1 -> 2, 2 -> 2, 3 -> 2, 4 -> 2)
val treeModel = GradientBoostedTrees.train(trainingData, boostingStrategy)
treeModel.predict(Vectors.dense(1,1,0,0,1)) // CNN,FB,NYTIMES
treeModel.predict(Vectors.dense(0,1,1,1,0)) // FB,IG,TWITTER
