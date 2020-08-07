package guardsuspension;

import java.util.LinkedList;
import java.util.stream.IntStream;

public class GuardSuspensionQueue {
    private final static LinkedList<Integer> queue = new LinkedList<>();
    private final int LIMIT = 100;

    public void offer(Integer data) throws InterruptedException {
        synchronized (this) {
            while (queue.size() >= LIMIT) {
                this.wait();
            }
            queue.addLast(data);
            this.notifyAll();
        }
    }

    public Integer take() throws InterruptedException {
        synchronized (this) {
            while (queue.size() <= 0) {
                this.wait();

            }
            this.notifyAll();
            return queue.removeFirst();
        }

    }

    public static void main(String[] args) {
        GuardSuspensionQueue guardsuspension = new GuardSuspensionQueue();
        //流线程读写
        while(true){
            IntStream.range(0, 5).forEach(i -> new Thread(() -> {
                try {
                    guardsuspension.offer(i * 10);
                    System.out.println(Thread.currentThread()+"-----"+i);
                } catch (InterruptedException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }).start());
        IntStream.range(0, 5).forEach(i -> new Thread(() -> {
            try {
                System.out.println(Thread.currentThread()+"-----"+guardsuspension.take());
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }).start());
        }
        

    }
}