package com.company.juc.jucdesign.activeobject;

import java.util.concurrent.TimeUnit;

import com.company.juc.jucdesign.future.Future;
import com.company.juc.jucdesign.future.FutureService;
public class OrderServiceImpl implements OrderService {

	@Override
	public Future<String> findOrferDetails(long orderId) {
		// TODO Auto-generated method stub
		return FutureService.<Long,String>newService().submit(input->{
            try {
                TimeUnit.SECONDS.sleep(10);
                System.out.println("process the orderID->"+orderId);
            } catch (InterruptedException e) {
                //TODO: handle exception
                e.printStackTrace();
            }
            return "The order Details Information";
        },orderId,null);
	}

	@Override
	public void order(String account, long orderId) {
        // TODO Auto-generated method stub
        try {
            TimeUnit.SECONDS.sleep(10);
            System.out.println("process the order for account"+account+",ordId"+orderId);
        } catch (InterruptedException e) {
            //TODO: handle exception
            e.printStackTrace();
        }
		
	}

}