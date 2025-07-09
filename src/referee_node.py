#!/usr/bin/env python3
# coding=utf-8

import rospy
from vision_pkg.msg import referee
import socket
import struct

class GameStateDecoder:
    def __init__(self):
        #initialize dictionaries used for decoding referee msgs
        self._penalty = {
            255:"unknown",
            0:"none",
            14:"substitute",
            30:"pushing",
            31:"ball_manipulation",
            34:"pickup/incapable",
            35:"service"
        }

        self._state = {
        -1: "quieto",
        0: "quieto",
        1: "Ready",
        2: "quieto",
        3: "Playing",
        4: "quieto"
        }    
        self._submode = {
            0: "still",
            1: "prepare",
            2: "still"
        }

        self._gameType = {
        -1: "Unknown",
        0: "round robin",
        1: "playoff",
        2: "drop-in"
        }

        self._secondaryState = {
        0: "Normal",
        1: "Penalty Shoot",
        2: "Overtime",
        3: "Timeout",
        4: "Direct Free Kick",
        5: "Indirect Free Kick",
        6: "Penalty Kick",
        7: "Corner Kick",
        8: "Goal Kick",
        9: "Throw-In"
        }
    def decode(self,rawData, refereeMsg, playerNumber):
        #first part
        protocolFirst8Bytes, protocolLast8Bytes, refereeMsg.packet_number, refereeMsg.players_per_team, gameType, state, refereeMsg.first_half, refereeMsg.kick_off_team, s_state, refereeMsg.teamPerformingSubMode, submode = struct.unpack('11B',rawData[4:15])
        #second part
        refereeMsg.drop_in_team, dropInTimeFirst8Bytes, dropInTimeLast8Bytes, timeFirs8Bytes, timeLast8Bytes, secondaryTimeFirst8Bytes, secondaryTimeLast8Bytes = struct.unpack('7B',rawData[17:24])
        

        if(refereeMsg.first_half==1): #los bytes cambian en el cambio de cancha local es 1, vistante 0
            teaminfo = rawData[24:29]
            playerinfo = rawData[290+playerNumber*6:296+playerNumber*6]
        else:
            teaminfo = rawData[356:361]
            playerinfo = rawData[622+playerNumber*6:628+playerNumber*6]

        #team info 356 to 361 for b team
        refereeMsg.teamNumber, refereeMsg.teamColor, refereeMsg.score, refereeMsg.penaltyShotCounter, refereeMsg.coachSequence = struct.unpack('5B',teaminfo)
        #player info 622+n*6:628 for b team 
        penalty, refereeMsg.penaltyTime, refereeMsg.warnings, refereeMsg.yellowCards, refereeMsg.redCards, refereeMsg.gkFlag = struct.unpack('6B',playerinfo)



        #16 bits values and dictionaries 
        refereeMsg.protocol_version = protocolFirst8Bytes | (protocolLast8Bytes<<8)
        refereeMsg.game_type = self._gameType.get(gameType,"unknown")
        refereeMsg.state = self._state.get(state)
        refereeMsg.secondary_state=self._secondaryState.get(s_state)
        refereeMsg.submode = self._submode.get(submode)
        refereeMsg.drop_in_time = dropInTimeFirst8Bytes | (dropInTimeLast8Bytes<<8)
        refereeMsg.secs_remaining = timeFirs8Bytes|(timeLast8Bytes<<8)
        refereeMsg.secondary_time = secondaryTimeFirst8Bytes|(secondaryTimeLast8Bytes<<8)
        refereeMsg.penalty=self._penalty.get(penalty)


        #important information about humanoid state of play
        if (refereeMsg.state == "quieto") | (refereeMsg.redCards == 1) | (refereeMsg.penaltyTime != 0) | (refereeMsg.secondary_state == "Timeout") | ((refereeMsg.submode == "still") & (refereeMsg.secondary_state !="Normal")):
            refereeMsg.robotPlayState = "quieto"
            refereeMsg.robotPlayStateInt = 0
            return refereeMsg
        if (refereeMsg.state == "Ready"):
            refereeMsg.robotPlayState = "acomodate"
            refereeMsg.robotPlayStateInt = 1
            return refereeMsg
        if (refereeMsg.state == "Playing") & (refereeMsg.secondary_state == "Normal") & (refereeMsg.redCards == 0) & (refereeMsg.penaltyTime == 0):
            refereeMsg.robotPlayState = "playing"
            refereeMsg.robotPlayStateInt = 2
            return refereeMsg
        if (refereeMsg.secondary_state != "Normal") & (refereeMsg.teamPerformingSubMode == 10) & (refereeMsg.submode != "still"):
            refereeMsg.robotPlayState = "acercate"
            refereeMsg.robotPlayStateInt = 3
            return refereeMsg
        if (refereeMsg.secondary_state != "Normal") & (refereeMsg.teamPerformingSubMode != 10) & (refereeMsg.submode != "still"):
            refereeMsg.robotPlayState = "alejate"
            refereeMsg.robotPlayStateInt = 4
            return refereeMsg
    

class RefereePublisher:
    def __init__(self, nodeName):
        self.robotID = rospy.get_param('robot_id', 0)
        rospy.init_node(f"nodeName_{self.robotID}", anonymous=True)
        #robot id starts from 0, but GameController starts from 1
        self._pub = rospy.Publisher(f'robotis_{self.robotID}/referee_data', referee, queue_size=1)
        self.refereeMsg = referee()
    def publish(self,msg):
        rospy.loginfo(f"publishing from robot_{self.robotID} which has ID={self.robotID}")
        self._pub.publish(msg)
        

class UDPCommunication:

    # Datos de cada jugador
    header = b'RGrt'    # Header RGrt
    version = 2         # Versión de la estructura de datos
    team = 14            # Número de equipo
    stdMsg = 2      # Mensaje (0: GAMECONTROLLER_RETURN_MSG_ALIVE, 1: GAMECONTROLLER_RETURN_MSG_MAN_PENALISE, 2: GAMECONTROLLER_RETURN_MSG_MAN_UNPENALISE)

    def __init__(self, publisherIp, refereeIp, listeningPort, sendingPort):
        self._ip = publisherIp
        self._refereeIp = refereeIp
        self._listeningPort = listeningPort
        self._sendingPort = sendingPort
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  #new datagram socket IPv4 
        self.returnSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self._ip,self._listeningPort))

    def listenReferee(self):
        data, addr = self.sock.recvfrom(1024) #1024 bytes
        return data
    def talk2Referee(self, msg):
        self.returnSock.sendto(msg, (self._refereeIp,self._sendingPort))


def main():
    udpHandler = UDPCommunication("192.168.0.255","192.168.0.236",3838,3939)#
    refPublisher = RefereePublisher("refereeNode")
    decoder = GameStateDecoder()
    msg2referee = struct.pack('<4s4B', UDPCommunication.header,UDPCommunication.version,UDPCommunication.team,refPublisher.robotID, UDPCommunication.stdMsg)      # Mensaje (0: GAMECONTROLLER_RETURN_MSG_ALIVE, 1: GAMECONTROLLER_RETURN_MSG_MAN_PENALISE, 2: GAMECONTROLLER_RETURN_MSG_MAN_UNPENALISE)
    while not rospy.is_shutdown():
        rawData =udpHandler.listenReferee()
        formattedMsg = decoder.decode(rawData,refPublisher.refereeMsg,refPublisher.robotID)
        udpHandler.talk2Referee(msg2referee)
        refPublisher.publish(formattedMsg)
if __name__ == '__main__':
    main()
