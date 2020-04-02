#include <iostream>
#include <string>
// #include <cctype>

using namespace std;



string to_upper(string str)
{
   string result = "";
   for (int i = 0; i < int(str.length()); i++)
   {
       // convert char to upper and concat to result
       result += toupper(str[i]);
   }
   return result;
}


void to_upper_inplace(string & str)
{
  for (int i = 0; i < int(str.length()); i++)
  {
      // convert char to upper through toupper() in <cctype>
      str[i] = toupper(str[i]);
  }
}


string acronym_of(string str)
{
    int space_loc = 0;
    string acronym(1, str[0]);  // string var_name(str_len, content)

    while (1)
    {
        space_loc = str.find(' ', space_loc + 1);

        if (space_loc == int(str.npos)) break;

        acronym += str[space_loc + 1];
    }

    return acronym;
}


int main()
{
  cout << "ACRONYM GENERATOR" << endl;

  while (1){


      string my_str;


      //my_str = "Hello How Are You?";

      // to_upper_case_inplace(my_str);
      // cout << my_str;

      //string ur_str = to_upper(my_str);
      //cout << ur_str;

      cout << "Enter your word: ";
      getline(cin, my_str);     // use getline(cin, var) to NOT skip the WHITESPACES
      cout << "The acronym of your word is: \"" << acronym_of(my_str) << "\".\n";

  }

}
