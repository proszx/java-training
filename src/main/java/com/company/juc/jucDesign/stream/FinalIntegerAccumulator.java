package stream;

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public final class FinalIntegerAccumulator {
    public final int init;

    public FinalIntegerAccumulator(int init) {
        this.init = init;
    }
    public FinalIntegerAccumulator(FinalIntegerAccumulator accumulator,int init){
        this.init=accumulator.getValue()+init;
    }

    public FinalIntegerAccumulator add(int i) {
        return new FinalIntegerAccumulator(this, i);
    }

    public int getValue() {
        return this.init;
    }

    public static void main(String[] args) {
        FinalIntegerAccumulator accumulator = new FinalIntegerAccumulator(0);
        IntStream.range(0, 3).forEach(i -> new Thread(() -> {
            int inc = 0;
            while (true) {
                
                //可见这里是非线程安全的 所以可以采用锁类的方式进行更改
                int oldValue = accumulator.getValue();
                int result = accumulator.add(inc).getValue();
                
                /**
                int oldValue;
                int result;
                synchronized(IntegerAccumulator.class){
                    oldValue = accumulator.getValue();
                    result = accumulator.add(inc);
                } */
                System.out.println(oldValue + "+" + inc + "=" + result);
                if (inc + oldValue != result) {
                    System.err.println("Error:" + oldValue + "+" + inc + "=" + result);
                }
                inc++;
                slowly(1);
            }
        }).start());
    }

    private static void slowly(int time) {
        try {
            TimeUnit.SECONDS.sleep(time);
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}