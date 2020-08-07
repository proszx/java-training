package twophase;
/** 
 * Thread-Pre-Msg模式
 * 为每个线程的处理开辟一个线程是的消息能够以并发的形式进行处理，提高吞吐能力
*/
public class Request {
    private final StringBuffer bus;
    public Request(StringBuffer bus){
        this.bus=bus;
    }
    @Override
    public String toString(){
        return bus.toString();
    }
}