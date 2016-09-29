package com.leetcode2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class WordDistance {
	HashMap<String, List<Integer>> wordIndices;
	public WordDistance(String[] words){
		wordIndices=new HashMap<String, List<Integer>>();
		for(int i=0;i<words.length;i++){
			String word=words[i];
			if(wordIndices.containsKey(word))
				wordIndices.get(word).add(i);
			else{
				List<Integer> lst=new ArrayList<Integer>();
				lst.add(i);
				wordIndices.put(word, lst);
			}
		}
	}
	
	public int shortest(String word1, String word2) {
		List<Integer> indices1=wordIndices.get(word1);
		List<Integer> indices2=wordIndices.get(word2);
		int i=0, j=0, minDis=Integer.MAX_VALUE;
		while(i<indices1.size()&&j<indices2.size()){
			int idx1=indices1.get(i);
			int idx2=indices2.get(j);
			minDis=Math.min(minDis, Math.abs(idx1-idx2));
			if(idx1<idx2)
				i++;
			else
				j++;
		}
		return minDis;
	}
}
