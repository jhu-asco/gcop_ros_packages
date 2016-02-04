/*
 * resourceStrParser.h
 *
 *  Created on: Dec 25, 2015
 *      Author: subhransu
 *  Copied this chunk of code from https://github.com/ros/resource_retriever.git
 */

#ifndef RESOURCE_URL_PARSER_H_
#define RESOURCE_URL_PARSER_H_
#include <string>
#include <ros/ros.h>
#include <ros/package.h>

using namespace std;

string getPathFromPkgUrl(string url)
{
  std::string mod_url = url;
  if (url.find("package://") == 0)
  {
    mod_url.erase(0, strlen("package://"));
    size_t pos = mod_url.find("/");
    if (pos == std::string::npos)
    {
      cout<<"Could not parse package:// format into file:// format"<<endl;
      return string();
    }

    std::string package = mod_url.substr(0, pos);
    mod_url.erase(0, pos);
    std::string package_path = ros::package::getPath(package);
    if (package_path.empty())
    {
      cout<<"Package [" + package + "] does not exist"<<endl;
      return string();
    }
    mod_url = package_path + mod_url;
    return mod_url;
  }
  else if (url.find("file://") == 0)
  {
    mod_url.erase(0, strlen("file://"));
    return mod_url;
  }
  else
  {
    cout<<"Doesn't contain the package:// tag in front of the path"<<endl;
    return string();
  }
}


#endif /* RESOURCE_URL_PARSER_H_ */
