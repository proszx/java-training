package latch;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

public class ProsTravel extends Thread{
    private final Latch latch;
    private final String pros;
    private final String trans;
    public ProsTravel(Latch latch,String pros,String trans){
        this.latch=latch;
        this.pros=pros;
        this.trans=trans;
    }
    @Override
    public void run(){
        System.out.println(pros+"start take"+trans+"]");
        try {
            TimeUnit.SECONDS.sleep(ThreadLocalRandom.current().nextInt(10));
        } catch (InterruptedException e) {
            e.printStackTrace();
            //TODO: handle exception
        }
        System.out.println(pros+"arrived"+ trans);
        latch.countDown();
    }
}