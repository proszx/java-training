package com.company.juc.jucdesign.activeobject;
import com.company.juc.jucdesign.future.Future;
public interface OrderService {
    Future<String> findOrferDetails(long orderId);
    void order(String account,long orderId);
}