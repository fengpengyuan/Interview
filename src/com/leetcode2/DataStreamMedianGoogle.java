package com.leetcode2;

//data stream of integer 0-999, write add() and median() function, makes them both run in O(1) time.
public class DataStreamMedianGoogle {
	int[] nums;
	int total;
	
	public DataStreamMedianGoogle(){
		this.nums=new int[1000];
		this.total=0;
	}
	
	public void add(int num){
		nums[num]++;
		total++;
	}
	
	public double median(){
		int count=0;
		int middle=(total+1)/2;

		for(int i=0;i<1000;i++){
			count+=nums[i];
			if(count>=middle){
				if(count==middle&&total%2==0)
					return (i+i+1)/2.0;
				return i;
			}
		}
		return  0;
	}
	
	public static void main(String[] args){
		DataStreamMedianGoogle dm=new DataStreamMedianGoogle();
		dm.add(1);
		System.out.println(dm.median());
        dm.add(2);
        System.out.println(dm.median());
        dm.add(3);
        System.out.println(dm.median());
        dm.add(3);
        System.out.println(dm.median());
        dm.add(3);
        System.out.println(dm.median());
	}
}
