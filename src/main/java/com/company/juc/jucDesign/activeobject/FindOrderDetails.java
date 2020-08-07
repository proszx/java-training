package com.company.juc.jucdesign.activeobject;

import java.util.Map;

import com.company.juc.jucdesign.future.Future;

public class FindOrderDetails extends MethodMessage{

    public FindOrderDetails(Map<String,Object> param,OrderService orderService){
        super(param,orderService);
    }
	@Override
	public void execute() {
	
        Future<String> real=orderService.findOrferDetails((Long)param.get("orderId"));
        ActiveFuture<String> activeFuture=(ActiveFuture<String>) param.get("activeFuture");
        
        try {
            String result=real.get();
            activeFuture.finish(result);
        } catch (InterruptedException e) {
            
            e.printStackTrace();
        }
	}

}
