package stream;

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class IntegerAccumulator {
    public int init;

    public IntegerAccumulator(int init) {
        this.init = init;
    }

    public int add(int i) {
        this.init += i;
        return this.init;
    }

    public int getValue() {
        return this.init;
    }

    public static void main(String[] args) {
        IntegerAccumulator accumulator = new IntegerAccumulator(0);
        IntStream.range(0, 3).forEach(i -> new Thread(() -> {
            int inc = 0;
            while (true) {
                /**
                //可见这里是非线程安全的 所以可以采用锁类的方式进行更改
                int oldValue = accumulator.getValue();
                int result = accumulator.add(inc);
                */
                int oldValue;
                int result;
                synchronized(IntegerAccumulator.class){
                    oldValue = accumulator.getValue();
                    result = accumulator.add(inc);
                }
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