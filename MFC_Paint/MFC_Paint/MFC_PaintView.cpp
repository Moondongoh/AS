/*
1. 윈도우 창 생성이 가능하다.
- 윈도우 창 타이틀 이름 수정 
- 윈도우 창 메뉴바 정리
2. 마우스 우 클릭을 이용해 선 그리기가 가능하다.
3. 마우스 좌 클릭을 이용해 지우기 기능이 가능하다.
4. 메뉴바 기능을 사용 가능하다.
- 새로 만들기, 새 창, 저장 및 열기
- 도형(직사각형, 원형)그리기가 가능하다.
- 선 굵기에 대해서 조정이 가능하다.
5. 도형 그리기 시 더블 버퍼링 작업이 된다.
6. 
*/


// MFC_PaintView.cpp: CMFCPaintView 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "MFC_Paint.h"
#endif

#include "MFC_PaintDoc.h"
#include "MFC_PaintView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMFCPaintView

IMPLEMENT_DYNCREATE(CMFCPaintView, CView)

// MFC_PaintView.cpp 파일에 다음과 같은 이벤트 처리기를 추가합니다.

BEGIN_MESSAGE_MAP(CMFCPaintView, CView)
    ON_WM_LBUTTONDOWN()
    ON_WM_MOUSEMOVE()
    ON_WM_LBUTTONUP()
    ON_WM_RBUTTONDOWN()
    ON_WM_RBUTTONUP()
    ON_COMMAND(ID_LINE_1, &CMFCPaintView::OnLine1)
    ON_COMMAND(ID_LINE_10, &CMFCPaintView::OnLine10)
    ON_COMMAND(ID_LINE_15, &CMFCPaintView::OnLine15)
END_MESSAGE_MAP()

void CMFCPaintView::OnLButtonDown(UINT nFlags, CPoint point)
{
    m_ptPrev = point; // 시작 점 설정
    m_bDrawing = true; // 드로잉 시작
    m_bErasing = false; // 지우개 모드 비활성화
    CView::OnLButtonDown(nFlags, point);
}

void CMFCPaintView::OnMouseMove(UINT nFlags, CPoint point)
{
    if (m_bDrawing)
    {
        // 좌표 저장할 때 CPoint 클래스 이용
        CDC* pDC = GetDC();
        CPen pen(PS_SOLID, m_nPenWidth, RGB(0, 0, 0)); // 선 굵기와 색상 설정
        CPen* pOldPen = pDC->SelectObject(&pen);

        pDC->MoveTo(m_ptPrev); // 이전 점에서
        pDC->LineTo(point); // 현재 점까지 선을 그림

        pDC->SelectObject(pOldPen);
        m_ptPrev = point; // 현재 점을 이전 점으로 업데이트
        ReleaseDC(pDC);
    }
    else if (m_bErasing)
    {
        CDC* pDC = GetDC();
        CPen pen(PS_SOLID, m_nPenWidth, RGB(255, 255, 255)); // 선 굵기와 색상 설정
        CPen* pOldPen = pDC->SelectObject(&pen);

        pDC->MoveTo(m_ptPrev); // 이전 점에서
        pDC->LineTo(point); // 현재 점까지 선을 그림

        pDC->SelectObject(pOldPen);
        m_ptPrev = point; // 현재 점을 이전 점으로 업데이트
        ReleaseDC(pDC);
    }

    CView::OnMouseMove(nFlags, point);
}

void CMFCPaintView::OnLButtonUp(UINT nFlags, CPoint point)
{
    if (m_bDrawing)
    {
        m_bDrawing = false; // 드로잉 끝
    }
    CView::OnLButtonUp(nFlags, point);
}

void CMFCPaintView::OnRButtonDown(UINT nFlags, CPoint point)
{
    m_ptPrev = point; // 시작 점 설정
    m_bErasing = true; // 지우개 시작
    m_bDrawing = false; // 드로잉 모드 비활성화
    CView::OnRButtonDown(nFlags, point);
}

void CMFCPaintView::OnRButtonUp(UINT nFlags, CPoint point)
{
    if (m_bErasing)
    {
        m_bErasing = false; // 지우개 끝
    }
    CView::OnRButtonUp(nFlags, point);
}

// CMFCPaintView 생성/소멸

CMFCPaintView::CMFCPaintView() noexcept
    : m_bDrawing(false), m_bErasing(false), m_nPenWidth(1) // 초기 선 굵기를 1로 설정
{
    // TODO: 여기에 생성 코드를 추가합니다.
}

CMFCPaintView::~CMFCPaintView()
{
}

BOOL CMFCPaintView::PreCreateWindow(CREATESTRUCT& cs)
{
    // 시작점
    m_bErasing = false; // 지우개 시작
    m_bDrawing = false; // 드로잉 모드 비활성화

	return CView::PreCreateWindow(cs);
}

// CMFCPaintView 그리기

void CMFCPaintView::OnDraw(CDC* /*pDC*/)
{
	CMFCPaintDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
}

// CMFCPaintView 진단

#ifdef _DEBUG
void CMFCPaintView::AssertValid() const
{
	CView::AssertValid();
}

void CMFCPaintView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMFCPaintDoc* CMFCPaintView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMFCPaintDoc)));
	return (CMFCPaintDoc*)m_pDocument;
}
#endif //_DEBUG


// CMFCPaintView 메시지 처리기


void CMFCPaintView::OnLine1()
{
    m_nPenWidth = 1; // 선 굵기를 1로 설정
}


void CMFCPaintView::OnLine10()
{
    m_nPenWidth = 10; // 선 굵기를 1로 설정
}


void CMFCPaintView::OnLine15()
{
    m_nPenWidth = 15; // 선 굵기를 1로 설정
}
