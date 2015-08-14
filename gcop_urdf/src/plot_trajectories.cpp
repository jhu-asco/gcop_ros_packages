/***************************************************************************
 * glut_example.cpp is part of Math Graphic Library
 * Copyright (C) 2007-2014 Alexey Balakin <mathgl.abalakin@gmail.ru>       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include <plplot/plstream.h>
#include <ros/ros.h>
#include <boost/thread.hpp>
#include <iostream>
#include "gcop_comm/CtrlTraj.h"

// -- Copied from plcdemoc --//
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <cstring>
#include <cmath>


using namespace std;

bool traj_recvd = false;
boost::thread *graph_thread;
ros::Subscriber traj_sub;
plstream *pls;
PLFLT *times;
vector<PLFLT*> controls;
vector<PLFLT*> maxandmincontrols;
PLFLT xmin = 0., xmax = 1., ymin = 0., ymax = 1.;
int color = 3;

//-----------------------------------------------------------------------------
void trajectory_Callback(const gcop_comm::CtrlTraj::ConstPtr& trajectory)
{
	if(!pls)
	{
		cout<<"Invalid pls pointer "<<endl;
		return;
	}
	//Print the graph:
	int usize = trajectory->ctrl[0].ctrlvec.size();
	int N = trajectory->N;
	cout<<"Control Vec size: "<<usize<<endl;
	if(!traj_recvd)
	{
		traj_recvd = true;
		times = (PLFLT *) calloc( N, sizeof ( PLFLT ) );
		controls.resize(usize); 
		maxandmincontrols.resize(usize);
		for(int count1 = 0; count1 < usize; count1++)
		{
			controls[count1] = (PLFLT *) calloc( N, sizeof ( PLFLT ) );
			maxandmincontrols[count1] = (PLFLT *) calloc( 2, sizeof(PLFLT) );
			maxandmincontrols[count1][0] = -1e6;//Max
			maxandmincontrols[count1][1] = 1e6;//Min
		}
		
		cout<<"Memory Allocation done "<<endl;
    //Denote number of subcolumns based on usize:
    if(usize%2 == 0)
      pls->star( 2, usize/2 );
    else
      pls->star( 2, usize/2+1 );
	}
	for(int count = 0;count < N; ++count)
	{
		times[count] = trajectory->time[count];
		cout<<count<<"\t"<<times[count]<<"\t";
		for(int count1 = 0;count1 < usize; count1++)
		{
			controls[count1][count]  = trajectory->ctrl[count].ctrlvec[count1];
			maxandmincontrols[count1][0]  = max(maxandmincontrols[count1][0],controls[count1][count]);
			maxandmincontrols[count1][1]  = min(maxandmincontrols[count1][1],controls[count1][count]);
			cout<<controls[count1][count]<<"\t";
		}
		cout<<endl;
	}
//	for(int count = 0;count < usize; count++)
//		pls->clear();
	color = (color+1)%3;
	for(int count = 0; count < usize; count++)
	{
		cout<<"MaxandMincontrols: ["<<count<<"]\t"<<maxandmincontrols[count][0]<<"\t"<<maxandmincontrols[count][1]<<endl;
		pls->adv(count+1);
		if((maxandmincontrols[count][0] - maxandmincontrols[count][1]) < 1e-2)
			maxandmincontrols[count][0] += 1e-2;
		/*if(count == 3)
			ymax = 20;
		else
			ymax = 0.5;
			*/
		pls->env0( times[0], times[N-1], maxandmincontrols[count][1], maxandmincontrols[count][0], 0, 0 );
		pls->col0(3);
		pls->line( N, times, controls[count]);
		pls->col0(1);
	}
	usleep(50000);
	//return 1;
}


int main(int argc,char **argv)
{
	ros::init(argc, argv,"plotter");
	ros::NodeHandle n;
	traj_sub = n.subscribe("/mbsddp/ctrltraj",1, trajectory_Callback);
	int argc_dummy = 3;
	const char* argv_dummy[3];
	argv_dummy[0] = "./exec";
	argv_dummy[1] = "-dev";
	argv_dummy[2] = "xwin";
	ROS_INFO("Argv: %s\t%s\t%s",argv_dummy[0], argv_dummy[1], argv_dummy[2]);
	pls = new plstream();
	//Initialize plot
	//const char **argc_dummy; 
	/*const char *arg1 = "exec";
	const char *arg2 = "-dev";
	const char *arg3 = "xwin";
	*/
	//= {"exec","-dev","xwin"};
  //sprintf(argc_dummy[0],"exec");
	//sprintf(argc_dummy[1],"-dev xwin");
	pls->parseopts( &argc_dummy, argv_dummy ,PL_PARSE_FULL);
//	pls->init();
	// Create a labelled box to hold the plot.
	ros::spin();
	return 0;
}
