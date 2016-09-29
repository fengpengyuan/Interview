package linkedIn;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public class TwoSumTest implements TwoSum2{
	//for O(1) test
	HashSet<Integer> sumSet;
	HashMap<Integer, Integer> map;
	public TwoSumTest(){
		map=new HashMap<Integer, Integer>();
		sumSet=new HashSet<Integer>();
	}
	 public void store(int input){
		 if(!map.containsKey(input))
			 map.put(input, 1);
		 else
			 map.put(input, map.get(input)+1);
	 }
	 
	 public boolean test(int val) {
		 for (int key : map.keySet()) {  
		        if (val - key == key) {  
		            if (map.get(key) >= 2) {  
		                return true;  
		            }  
		        } else if (map.containsKey(val - key)) {  
		            return true;  
		        }  
		    }  
		  
		    return false;  	 
		 
//		 Iterator<Integer> it=map.keySet().iterator();
//		 while(it.hasNext()){
//			 int key=it.next();
//			 if(map.containsKey(val-key)){
//				 boolean isDouble=val==2*key;
//				 int count=map.get(key);
//				 if(!(isDouble&&count==1))
//					 return true;	
//			 }
//		 }
//		 return false;
	 }
	 
	 
//	 follow up, make test is O(1) time?
	 
	 public void store2(int input){
		 if(!map.containsKey(input))
			 map.put(input, 1);
		 else
			 map.put(input, map.get(input)+1);
		 Iterator<Integer> it=map.keySet().iterator();
		 while(it.hasNext()){
			 int key=it.next();
			 int sum=key+input;
			 sumSet.add(sum);
		 }
	 }
	 public boolean test2(int val){
		 return sumSet.contains(val);
	 }
}
