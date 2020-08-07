package com.company.juc.jucdesign.activeobject;

import com.company.juc.jucdesign.future.Future;
import java.util.HashMap;
import java.util.Map;
public class OrderServiceProxy implements OrderService{
    private final OrderService orderService;
    private final ActiveMessageQueue activeQueue;

    public OrderServiceProxy(OrderService orderService,ActiveMessageQueue activeQueue){
        this.activeQueue=activeQueue;
        this.orderService=orderService;
    }
	@Override
	public Future<String> findOrferDetails(long orderId) {
		// TODO Auto-generated method stub
        //return null;
        final ActiveFuture<String> activeFuture=new ActiveFuture<>();
        Map<String,Object> param=new HashMap<>();
        param.put("orderId", orderId);
        param.put("activeFuture", activeFuture);
        MethodMessage msg=new FindOrderDetails(param,orderService);
        activeQueue.offer(msg);
        return activeFuture;
	}

	@Override
	public void order(String account, long orderId) {
        // TODO Auto-generated method stub
        Map<String,Object> param=new HashMap<>();
        param.put("account", account);
        param.put("orderId", orderId);
        MethodMessage msg=new OrderMessage(param,orderService);
        activeQueue.offer(msg);
		
	}

}