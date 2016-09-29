package com.leetcode2;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class ValidWordAbbr {
	HashMap<String, Set<String>> map=new HashMap<String, Set<String>>();
	public ValidWordAbbr(String[] dictionary) {  
		for(String word: dictionary){
			String s=word;
			if(s.length()>2)
				s=s.charAt(0)+""+(s.length()-2)+s.charAt(s.length()-1);
			if(map.containsKey(s)){
				map.get(s).add(word);
			}else{
				Set<String> set=new HashSet<String>();
				set.add(word);
				map.put(s, set);
			}
		}
	}
	
	public boolean isUnique(String word) {  
		if(word.length()>2)
			word=word.charAt(0)+""+(word.length()-2)+word.charAt(word.length()-1);
		if(!map.containsKey(word))
			return true;
		return map.containsKey(word)&&map.get(word).size()<2;
	}
}
