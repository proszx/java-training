package com.company.juc.jucdesign.activeobject;

import java.util.Map;

public class OrderMessage extends MethodMessage{
    public OrderMessage(Map<String,Object> param,OrderService orderService){
        super(param,orderService);
    }
	@Override
	public void execute() {
		// TODO Auto-generated method stub
        String account=(String)param.get("account");
        long orderId=(long)(param.get("orderId"));

        orderService.order(account, orderId);
	}

}
