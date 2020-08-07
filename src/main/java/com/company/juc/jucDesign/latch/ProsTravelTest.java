package latch;

import java.util.concurrent.TimeUnit;

public class ProsTravelTest {
    public static void main(String[] args) throws InterruptedException {
        Latch latch = new CountDownLatch(4);
        new ProsTravel(latch,"Alex","Bus").start();
        new ProsTravel(latch,"Alexd","Busi").start();
        new ProsTravel(latch,"Alexs","Busa").start();
        new ProsTravel(latch,"Alexa","Busw").start();
        //latch.await();
        try {
			latch.await(TimeUnit.SECONDS, 5);
		} catch (WaitTimeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        
        System.out.println("all arrived");
    }
}