package com.company.juc.thread;

import java.util.LinkedHashMap;

public class extentThread  extends LinkedHashMap implements  Runnable {

    @Override
    public void run() {
        System.out.println("run as runnable");
    }
}
