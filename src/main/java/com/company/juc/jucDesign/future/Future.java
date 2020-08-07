package  com.company.juc.jucdesign.future;
public interface Future<T> {
    T get() throws InterruptedException;
    boolean done();
}