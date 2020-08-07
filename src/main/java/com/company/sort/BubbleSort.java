package com.company.sort;

public class BubbleSort<T extends Comparable<T>> extends  Func<T>{
    @Override
    public void sort(T[] nums) {
        boolean isSorted=false;
        for(int i=nums.length-1;i>=0&&!isSorted;i--){
            isSorted=true;
            for(int j=0;j<i;j++){
                if(less(nums[j+1],nums[j])){
                    isSorted=false;
                    swap(nums,j+1,j);
                }
            }
        }
    }

}
