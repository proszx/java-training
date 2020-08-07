package com.company.sort;

public abstract class Func<T extends  Comparable<T>> {
    protected  void swap(T[] a,int i,int j){
        T tmp=a[i];
        a[i]=a[j];
        a[j]=tmp;
    }
    protected  boolean less(T c,T v){
        return c.compareTo(v)<0;
    }
    public  abstract  void sort(T[] nums);

}
