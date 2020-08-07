package com.company.juc.jucdesign.balking;
public class BalkingTest {
    public static void main(String[] args) {
        new DocumentEditThread("D:\\","test.txt").start();
    }
}