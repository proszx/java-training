package com.company.sort;

public class SelectionSort<T>  extends  Func{
    @Override
    public void sort(Comparable[] nums) {
        int N=nums.length;
        for(int i=0;i<N-1;i++){
            int min=i;
            for(int j=i+1;j<N;j++){
                if(less(nums[min],nums[j])){
                    min=j;
                }
            }
            swap(nums,i,min);
        }
    }
}
