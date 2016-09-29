package linkedIn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class WordDistance {
	HashMap<String, List<Integer>> map=new HashMap<String, List<Integer>>();
	public WordDistance(String[] words) {
		for(int i=0;i<words.length;i++){
			String word=words[i];
			if(map.containsKey(word))
				map.get(word).add(i);
			else{
				List<Integer> idx=new ArrayList<Integer>();
				idx.add(i);
				map.put(word, idx);
			}
		}
	}
	
	public int shortest(String word1, String word2) {
		List<Integer> idx1=map.get(word1);
		List<Integer> idx2=map.get(word2);
		
		int i=0, j=0;
		int minDis=Integer.MAX_VALUE;
		while(i<idx1.size()&&j<idx2.size()){
			minDis=Math.min(minDis, Math.abs(idx1.get(i)-idx2.get(j)));
			if(idx1.get(i)<idx2.get(j))
				i++;
			else
				j++;
		}
		return minDis;
	}
}
