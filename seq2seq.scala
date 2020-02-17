//report
//系列変換モデル

import CLASS._
import breeze.linalg._
import scala.sys.process._



object seq2seq{

  val dn = 100
  val ln = 100
  val target = io.Source.fromFile("/home/share/seq2seq.txt").getLines.toArray


  ////////////////各行をタプルに->単語をベクトルに////////////////////////////
  def load(fn:String)={
    var x = List[Array[Array[Double]]]()
    var n = List[Array[Array[Double]]]()
    val file = io.Source.fromFile(fn).getLines.toArray
    for(i<-0 until file.size){
      if(i%2 == 0){
        x ::= file(i).map(conv).toArray
      }else{
        n ::= file(i).map(conv).toArray
      }
    }

      (x.reverse).zip(n.reverse)
  }

  ////////////////単語をベクトルに//////////////////////////////////////
  def conv(c:Char)={
    val x = DenseVector.zeros[Double](target.size)
    x(target.indexOf(c)) = 1d
    x.toArray
  }


  /////////////main/////////////////////////////////////////////////
  def main(){

    //ネットワーク構成
    val N = new network()
    val lstm = new LSTM(100,100)
    val encode = List(
      new Affine(100,100),
      new LSTM(100,100)
    )
    val decode = List(
      new Affine(100,100),
      lstm,
      new Affine(100,100)
    )

    //load
    val dtrain = load("/home/share/sequence.kana.txt")
   
    //learning
    for(i<-0 until ln){
      print(i+"-->")
      var correct = 0d
      var err = 0d
      var y = new Array[Double](target.size)
      var z = new Array[Double](target.size)

      var count = 0
      for((x,n)<-dtrain.take(dn)){

        //encode<forward>
        for(p<-0 until x.size){
          z = N.forwards(encode,x(p).toArray)
        }

        lstm.setstate(DenseVector(z))

        //decode<forward>
        for(q<-0 until n.size){
          y = N.forwards(decode,y)
        }

        //計算
        if(argmax(y) == n(count)){
          correct += 1
        }
       // err += -math.log(y(argmax(n)))


        //decode<backward>
        for(p<-0 until n.size){
          //N.backwards(decode.reverse,y(p)-n(p))
        }

        //encode<backward>
        for(q<-0 until x.size){
        }




      }//dn
    }//ln
  }//main
}//object
 

