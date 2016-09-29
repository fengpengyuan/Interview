package com.leetcode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class WordDistance {
	Map<String, List<Integer>> wordIndexes=new HashMap<String, List<Integer>>();
	
	public WordDistance(String[] words){
		for(int i=0;i<words.length;i++){
			String word=words[i];
			if(wordIndexes.containsKey(word))
				wordIndexes.get(word).add(i);
			else{
				List<Integer> indexes=new ArrayList<Integer>();
				indexes.add(i);
				wordIndexes.put(word, indexes);
			}
		}
	}
	
	public int shortest(String word1, String word2) {
		List<Integer> index_word1=wordIndexes.get(word1);
		List<Integer> index_word2=wordIndexes.get(word2);
		
		int minDist=Integer.MAX_VALUE;
		int i=0,j=0;
		
		while(i<index_word1.size()&&j<index_word2.size()){
			minDist=Math.min(minDist, Math.abs(index_word1.get(i)-index_word2.get(j)));
			if(index_word1.get(i)<index_word2.get(j))
				i++;
			else
				j++;
		}
		return minDist;
	}
}
