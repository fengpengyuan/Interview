package others;
//zenefits
public class Trie {
	TrieNode root;
	public Trie(){
		root=new TrieNode(' ');
	}
	
	public void insert(String s){
		TrieNode node=root;
		for(int i=0;i<s.length();i++){
			TrieNode[] children=node.children;
			char c=s.charAt(i);
			if(children[c-'a']==null){
				children[c-'a']=new TrieNode(c);
			}
			else
				children[c-'a'].count++;
			node=children[c-'a'];
		}
		node.isComplete=true;
	}
	
	public int search(String s){
		if(s.length()==0){
			System.out.println("0");
			return 0;
		}
		TrieNode node=root;
		for(int i=0;i<s.length();i++){
			char c=s.charAt(i);
			TrieNode[] children=node.children;
			if(children[c-'a']==null){
				System.out.println("0");
				return 0;
			}
			else
				System.out.println(children[c-'a'].count);
			node=children[c-'a'];
		}
		return node.count;
	}
	
	public static void main(String[] args){
		Trie tree=new Trie();
		tree.insert("a");
		tree.insert("apple");
		tree.insert("apply");
		tree.insert("appreciate");
		tree.insert("book");
		tree.insert("beats");
		
		tree.search("appl");
	}
}
