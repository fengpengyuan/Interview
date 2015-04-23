package facebook;

import java.util.HashMap;


//Longest prefix matching – A Trie based solution in Java
//Given a dictionary of words and an input string, find the longest prefix of the string which is also a word in dictionary.

public class Trie {
	TrieNode root;
	public Trie(){
		root=new TrieNode(' ');
	}
	
	public void insert(String word){
		TrieNode node=root;
		for(int i=0;i<word.length();i++){
			HashMap<Character, TrieNode> children=node.getChildren();
			char c=word.charAt(i);
			if(children.containsKey(c))
				node=children.get(c);
			else{
				TrieNode t=new TrieNode(c);
				children.put(c, t);
				node=t;
			}
		}
		node.setIsEnd(true);
	}
	
	public String getMatchingPrefix(String input)  {
		String res="";
		int preMatch=0;
		TrieNode crawl=root;
		
		for(int i=0;i<input.length();i++){
			char c=input.charAt(i);
			HashMap<Character, TrieNode> children=crawl.getChildren();
			if(children.containsKey(c)){
				res+=c;
				crawl=children.get(c);
				
				if(crawl.isEnd()){
					preMatch=i+1;
				}
			}
			else
				break;
		}
		// If the last processed character did not match end of a word, 
        // return the previously matching prefix		
		if(!crawl.isEnd())
			return input.substring(0,preMatch);
		return res;
		
	}
	
	public static void main(String[] args){
		Trie dict = new Trie();        
        dict.insert("are");
        dict.insert("area");
        dict.insert("base");
        dict.insert("cat");
        dict.insert("cater");        
        dict.insert("basement");
         
        String input = "caterer";
        System.out.print(input + ": ");
        System.out.println(dict.getMatchingPrefix(input));
        
        input = "basement";
        System.out.print(input + ":   ");
        System.out.println(dict.getMatchingPrefix(input));                      
         
        input = "are";
        System.out.print(input + ":   ");
        System.out.println(dict.getMatchingPrefix(input));              
 
        input = "arex";
        System.out.print(input + ":   ");
        System.out.println(dict.getMatchingPrefix(input));              
 
        input = "basemexz";
        System.out.print(input + ":   ");
        System.out.println(dict.getMatchingPrefix(input));                      
         
        input = "xyz";
        System.out.print(input + ":   ");
        System.out.println(dict.getMatchingPrefix(input)); 
	}
}
