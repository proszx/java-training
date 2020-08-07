package workerthread;

import java.util.Random;
import java.util.concurrent.TimeUnit;

public class Worker  extends Thread{
    private final ProdChannel channel;
    private final static Random random=new Random(System.currentTimeMillis());

    public Worker(String workerName,ProdChannel channel){
        super(workerName);
        this.channel=channel;
    }
    @Override
    public void run(){
        while(true){
            try {
                Production prods=channel.takeProd();
                System.out.println(getName()+"execute"+prods);
                prods.create();
                TimeUnit.SECONDS.sleep(random.nextInt(10));
            } catch (InterruptedException e) {
                //TODO: handle exception
                e.printStackTrace();
            }
        }
    }
}
