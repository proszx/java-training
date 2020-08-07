package twophase;

import java.io.IOException;
import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.net.Socket;

public class SocketCleaningTracker {
    private final static ReferenceQueue<Object> queue=new ReferenceQueue<>();
    static {
        new Cleaner().start();
    }
    public static void track(Socket socket){
        new Tracker(socket,queue);
    }
    private static class Cleaner extends Thread{
        private Cleaner(){
            super("Socket SocketCleaningTracker");
            setDaemon(true);
        }
        @Override
        public void run(){
            while(true){
                try {
                    Tracker tracker=(Tracker)queue.remove();
                    tracker.close();
                } catch (InterruptedException e) {
                    //TODO: handle exception
                }
            }
        }
    }
    private static class Tracker extends PhantomReference<Object>{
        private final Socket socket;
        Tracker(Socket socket,ReferenceQueue<? super Object> queue){
            super(socket, queue);
            this.socket=socket;
        }
        public void close(){
            try {
                socket.close();
            } catch (IOException e) {
                //TODO: handle exception
                e.printStackTrace();
            }
        }
    }
}