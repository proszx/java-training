package workerthread;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import static java.util.concurrent.ThreadLocalRandom.current;

import java.util.concurrent.TimeUnit;
public class WorkerThreadTest {
    public static void main(String[] args) {
        final ProdChannel channel=new ProdChannel(5);
        AtomicInteger prodNo=new AtomicInteger();
        IntStream.range(0,9).forEach(i->new Thread(()->{
            while(true){
                channel.offerProduction(new Production(prodNo.getAndIncrement()));
                try {
                    TimeUnit.SECONDS.sleep(current().nextInt(10));
                } catch (InterruptedException e) {
                    //TODO: handle exception
                    e.printStackTrace();
                }
            }
        }).start());

    }
}